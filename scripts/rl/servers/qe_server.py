#!/usr/bin/env python3
"""Serve COMET QE scores with an optional OpenLID language-gated endpoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import uvicorn
from comet import load_from_checkpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .openlid_gate import OpenLIDGate
except ImportError:
    from openlid_gate import OpenLIDGate


LOGGER = logging.getLogger("qe_server")


class TranslationItem(BaseModel):
    src: str
    mt: str
    ref: str | None = None


class ScoreRequest(BaseModel):
    items: list[TranslationItem]


class OpenLIDItem(BaseModel):
    src: str
    mt: str
    tgt_lang: str | None = None


class OpenLIDRequest(BaseModel):
    items: list[OpenLIDItem]


def resolve_checkpoint(model_path: str) -> str:
    root = Path(model_path)
    if root.is_file():
        return str(root)
    if not root.is_dir():
        raise FileNotFoundError(f"COMET model path not found: {root}")
    default_checkpoint = root / "checkpoints" / "model.ckpt"
    if default_checkpoint.is_file():
        return str(default_checkpoint)
    matches = sorted(root.rglob("*.ckpt"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Could not find a .ckpt under {root}")


def configure_hf_cache(hf_cache: str | None, set_hub_cache: bool) -> None:
    if not hf_cache:
        return
    cache = str(Path(hf_cache))
    os.environ.setdefault("HF_HOME", cache)
    if set_hub_cache:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache)


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return to_jsonable(vars(value))
    return value


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def prediction_get(prediction: Any, key: str, default: Any = None) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


class RequestBatcher:
    """Combine concurrent HTTP requests before running a synchronous scorer."""

    def __init__(
        self,
        process: Callable[[list[dict[str, str]]], list[dict[str, Any]]],
        *,
        name: str,
        max_wait_ms: int,
        max_batch_size: int,
        slow_request_ms: int,
    ):
        self.process = process
        self.name = name
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.slow_request_ms = slow_request_ms
        self.pending: list[tuple[list[dict[str, str]], asyncio.Future, float]] = []
        self.pending_lock = asyncio.Lock()
        self.process_lock = asyncio.Lock()
        self.flush_task: asyncio.Task | None = None
        self.request_total = 0
        self.item_total = 0
        self.flush_total = 0
        self.slow_total = 0
        self.max_pending_depth = 0
        self.max_queue_wait = 0.0

    async def score(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        if not items:
            return []
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        enqueued_at = time.perf_counter()
        async with self.pending_lock:
            self.pending.append((items, future, enqueued_at))
            total = sum(len(batch) for batch, _, _ in self.pending)
            self.max_pending_depth = max(self.max_pending_depth, len(self.pending))
            if self.flush_task is None or self.flush_task.done():
                self.flush_task = asyncio.create_task(self._delayed_flush())
            if total >= self.max_batch_size:
                asyncio.create_task(self._flush())

        result = await future
        elapsed_ms = (time.perf_counter() - enqueued_at) * 1000
        if elapsed_ms >= self.slow_request_ms:
            self.slow_total += 1
            LOGGER.warning(
                "Slow %s request: %d item(s) took %.0fms (slow_total=%d)",
                self.name,
                len(items),
                elapsed_ms,
                self.slow_total,
            )
        return result

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._flush()

    async def _flush(self) -> None:
        async with self.pending_lock:
            pending = self.pending
            self.pending = []
        if not pending:
            return

        flat_items = [item for batch, _, _ in pending for item in batch]
        started = time.perf_counter()
        queue_wait = max(started - enqueued_at for _, _, enqueued_at in pending)
        self.max_queue_wait = max(self.max_queue_wait, queue_wait)
        try:
            async with self.process_lock:
                loop = asyncio.get_running_loop()
                flat_results = await loop.run_in_executor(None, self.process, flat_items)
            if len(flat_results) != len(flat_items):
                raise RuntimeError(
                    f"{self.name} returned {len(flat_results)} results for {len(flat_items)} items"
                )
        except Exception as error:
            LOGGER.exception("%s failed for %d item(s)", self.name, len(flat_items))
            for _, future, _ in pending:
                if not future.done():
                    future.set_exception(error)
            return

        elapsed = time.perf_counter() - started
        self.request_total += len(pending)
        self.item_total += len(flat_items)
        self.flush_total += 1
        LOGGER.info(
            "%s scored %d item(s) from %d request(s) in %.3fs (%.1f item/s)",
            self.name,
            len(flat_items),
            len(pending),
            elapsed,
            len(flat_items) / elapsed if elapsed > 0 else float("inf"),
        )

        cursor = 0
        for batch, future, _ in pending:
            size = len(batch)
            if not future.done():
                future.set_result(flat_results[cursor : cursor + size])
            cursor += size

        async with self.pending_lock:
            if self.pending and (
                self.flush_task is None
                or self.flush_task.done()
                or self.flush_task is asyncio.current_task()
            ):
                self.flush_task = asyncio.create_task(self._delayed_flush())

    def stats(self) -> dict[str, Any]:
        return {
            "flushes": self.flush_total,
            "requests": self.request_total,
            "items": self.item_total,
            "slow_requests": self.slow_total,
            "max_pending_depth": self.max_pending_depth,
            "max_queue_wait_s": round(self.max_queue_wait, 3),
        }


class CometPredictor:
    """Run direct COMET ``predict_step`` inference in bounded GPU batches."""

    def __init__(self, model: Any, device: torch.device, predict_batch_size: int):
        self.model = model
        self.device = device
        self.predict_batch_size = predict_batch_size

    def __call__(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(items), self.predict_batch_size):
            batch_items = items[start : start + self.predict_batch_size]
            prepared = self.model.prepare_for_inference(batch_items)
            prepared = move_to_device(prepared, self.device)
            with torch.inference_mode():
                prediction = self.model.predict_step(
                    prepared,
                    batch_idx=start // self.predict_batch_size,
                )

            scores = to_jsonable(prediction_get(prediction, "scores", []))
            metadata = prediction_get(prediction, "metadata", {})
            error_spans = to_jsonable(
                prediction_get(metadata, "error_spans", [[] for _ in scores])
            )
            if not isinstance(error_spans, list) or len(error_spans) != len(scores):
                error_spans = [[] for _ in scores]
            for index, score in enumerate(scores):
                results.append(
                    {
                        "score": float(score),
                        "error_spans": error_spans[index],
                    }
                )
        return results


def build_app(
    predictor: CometPredictor,
    *,
    checkpoint: str,
    device: str,
    max_wait_ms: int,
    max_server_batch_size: int,
    slow_request_ms: int,
    openlid_gate: OpenLIDGate | None = None,
    openlid_max_wait_ms: int = 20,
    openlid_max_batch_size: int = 64,
) -> FastAPI:
    score_batcher = RequestBatcher(
        predictor,
        name="COMET QE",
        max_wait_ms=max_wait_ms,
        max_batch_size=max_server_batch_size,
        slow_request_ms=slow_request_ms,
    )
    openlid_batcher = None
    if openlid_gate is not None:

        def evaluate_openlid(items: list[dict[str, str]]) -> list[dict[str, Any]]:
            return openlid_gate.evaluate_batch(
                [item.get("mt", "") for item in items],
                [item.get("tgt_lang") for item in items],
            )

        openlid_batcher = RequestBatcher(
            evaluate_openlid,
            name="OpenLID",
            max_wait_ms=openlid_max_wait_ms,
            max_batch_size=openlid_max_batch_size,
            slow_request_ms=slow_request_ms,
        )

    app = FastAPI(title="COMET QE reward server")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "checkpoint": checkpoint,
            "device": device,
            "openlid": openlid_batcher is not None,
            "score_stats": score_batcher.stats(),
            "openlid_stats": openlid_batcher.stats() if openlid_batcher else None,
        }

    @app.post("/score")
    async def score(request: ScoreRequest) -> dict[str, Any]:
        try:
            items = [item.model_dump(exclude_none=True) for item in request.items]
            return {"results": await score_batcher.score(items)}
        except Exception as error:
            LOGGER.exception("COMET /score failed for %d item(s)", len(request.items))
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.post("/score_openlid")
    async def score_openlid(request: OpenLIDRequest) -> dict[str, Any]:
        if openlid_batcher is None:
            raise HTTPException(status_code=400, detail="OpenLID is not enabled on this server")
        try:
            items = [item.model_dump() for item in request.items]
            qe_items = [{"src": item["src"], "mt": item["mt"]} for item in items]
            qe_results, openlid_results = await asyncio.gather(
                score_batcher.score(qe_items),
                openlid_batcher.score(items),
            )
            results = []
            for qe_result, openlid_result in zip(
                qe_results,
                openlid_results,
                strict=True,
            ):
                results.append(
                    {
                        "qe": float(qe_result.get("score", 0.0)),
                        "wa": 0.0,
                        "la_ok": int(openlid_result["la_ok"]),
                        "la_skip": int(openlid_result["la_skip"]),
                        "wa_precision": 0.0,
                        "wa_recall": 0.0,
                        "pred_iso": openlid_result.get("pred_iso", ""),
                        "tgt_iso": openlid_result.get("tgt_iso", ""),
                        "error_spans": qe_result.get("error_spans", []),
                    }
                )
            return {"results": results}
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("COMET /score_openlid failed for %d item(s)", len(request.items))
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app


def create_app(args: argparse.Namespace) -> FastAPI:
    checkpoint = resolve_checkpoint(args.model)
    configure_hf_cache(args.hf_cache, set_hub_cache=args.hf_hub_cache)
    device = torch.device(args.device)

    LOGGER.info("Loading COMET checkpoint: %s", checkpoint)
    model = load_from_checkpoint(checkpoint, local_files_only=args.local_files_only)
    model.eval()
    model.to(device)
    if hasattr(model, "input_weights_spans") and isinstance(
        model.input_weights_spans,
        torch.Tensor,
    ):
        model.input_weights_spans = model.input_weights_spans.to(device)

    predictor = CometPredictor(model, device, args.predict_batch_size)
    openlid_gate = OpenLIDGate(args.openlid_model) if args.openlid_model else None
    return build_app(
        predictor,
        checkpoint=checkpoint,
        device=str(device),
        max_wait_ms=args.max_wait_ms,
        max_server_batch_size=args.max_server_batch_size,
        slow_request_ms=args.slow_request_ms,
        openlid_gate=openlid_gate,
        openlid_max_wait_ms=args.openlid_max_wait_ms,
        openlid_max_batch_size=args.openlid_max_batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve COMET QE scores with an optional OpenLID gate."
    )
    parser.add_argument("--model", required=True, help="local COMET checkpoint file or model directory")
    parser.add_argument("--openlid-model", help="local OpenLID-v3 fastText model")
    parser.add_argument("--hf-cache", default=os.getenv("HF_HOME"))
    parser.add_argument(
        "--hf-hub-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also set HUGGINGFACE_HUB_CACHE to --hf-cache",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--max-server-batch-size", type=int, default=32)
    parser.add_argument("--max-wait-ms", type=int, default=50)
    parser.add_argument("--openlid-max-wait-ms", type=int, default=20)
    parser.add_argument("--openlid-max-batch-size", type=int, default=64)
    parser.add_argument("--slow-request-ms", type=int, default=3000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("high")
    uvicorn.run(
        create_app(args),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
