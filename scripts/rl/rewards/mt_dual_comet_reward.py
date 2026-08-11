"""Dual-QE sequence reward with an OpenLID language gate.

The xCOMET/OpenLID endpoint returns a reference-free xCOMET QE score and a
language-match decision. The CometKiwi endpoint returns a second
reference-free QE score. Language-correct outputs receive the mean of the two
QE scores; language-mismatched outputs receive ``la_fail_score``.

Both services are external HTTP dependencies. The client does not load reward
models locally.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
from pathlib import Path
from typing import Any

# Reuse the HTTP client helpers from the adjacent xCOMET reward module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt_xcomet_reward import (  # noqa: E402
    _extract_source,
    _float_arg,
    _int_arg,
    _normalize,
    _xcomet_post,
)

DEFAULT_XCOMET_URL = "http://127.0.0.1:8008/score_openlid"
DEFAULT_COMETKIWI_URL = "http://127.0.0.1:8012/score"


def _parse_urls_named(urls: Any, url: Any, env_urls: str, env_url: str, default: str) -> list[str]:
    """Resolve comma-separated service URLs from arguments or environment."""
    raw = urls or os.getenv(env_urls) or url or os.getenv(env_url) or default
    if isinstance(raw, (list, tuple)):
        parsed = [str(x).strip() for x in raw if str(x).strip()]
    else:
        parsed = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not parsed:
        raise ValueError(f"At least one URL is required (env {env_urls}/{env_url}).")
    return parsed


def _score_batch(
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    *,
    xcomet_urls: list[str],
    cometkiwi_urls: list[str],
    timeout_s: float,
    retries: int,
    la_fail_score: float,
) -> list[dict[str, Any]]:
    """Score a batch while preserving input order."""
    openlid_items: list[dict[str, str]] = []
    cometkiwi_items: list[dict[str, str]] = []
    for solution, _ground_truth, extra_info in zip(
        solution_strs,
        ground_truths,
        extra_infos,
        strict=True,
    ):
        extra_info = extra_info or {}
        src = _extract_source(extra_info)
        mt = _normalize(solution)
        tgt_lang = str(extra_info.get("target_language") or "")
        openlid_items.append({"src": src, "mt": mt, "tgt_lang": tgt_lang})
        cometkiwi_items.append({"src": src, "mt": mt})

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_x = pool.submit(_xcomet_post, openlid_items, xcomet_urls, timeout_s, retries)
        fut_k = pool.submit(_xcomet_post, cometkiwi_items, cometkiwi_urls, timeout_s, retries)
        openlid_results = fut_x.result()
        cometkiwi_results = fut_k.result()

    n = len(solution_strs)
    if not (len(openlid_results) == len(cometkiwi_results) == n):
        raise RuntimeError(
            f"dualcomet result length mismatch: xcomet_openlid={len(openlid_results)} "
            f"cometkiwi={len(cometkiwi_results)} expected={n}"
        )

    scores: list[dict[str, Any]] = []
    for xres, kres in zip(openlid_results, cometkiwi_results, strict=True):
        xcomet = float(xres.get("qe", 0.0))
        cometkiwi = float(kres.get("score", 0.0))
        la_ok = int(xres.get("la_ok", 0))
        if la_ok == 0:
            score = float(la_fail_score)
        else:
            score = (xcomet + cometkiwi) / 2.0
        scores.append(
            {
                "score": float(score),
                "xcomet": xcomet,
                "cometkiwi": cometkiwi,
                "la_ok": float(la_ok),
                "la_skip": float(xres.get("la_skip", 0)),
                "pred_iso": xres.get("pred_iso", ""),
                "tgt_iso": xres.get("tgt_iso", ""),
            }
        )
    return scores


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos: list[dict[str, Any]] | None = None,
    xcomet_url: str | None = None,
    xcomet_urls: str | list[str] | None = None,
    cometkiwi_url: str | None = None,
    cometkiwi_urls: str | list[str] | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
    la_fail_score: float | None = None,
    **_: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """verl custom reward entry point for batch or single-example calls.

    Explicit function arguments take precedence over environment variables and
    built-in defaults. ``DUALCOMET_TIMEOUT_S`` and ``DUALCOMET_RETRIES`` can
    override their corresponding ``XCOMET_*`` values.
    """
    resolved_xcomet = _parse_urls_named(
        xcomet_urls,
        xcomet_url,
        "XCOMET_URLS",
        "XCOMET_URL",
        DEFAULT_XCOMET_URL,
    )
    resolved_cometkiwi = _parse_urls_named(
        cometkiwi_urls,
        cometkiwi_url,
        "COMETKIWI_URLS",
        "COMETKIWI_URL",
        DEFAULT_COMETKIWI_URL,
    )
    resolved_timeout_s = _float_arg(timeout_s, "DUALCOMET_TIMEOUT_S", _float_arg(None, "XCOMET_TIMEOUT_S", 120.0))
    resolved_retries = _int_arg(retries, "DUALCOMET_RETRIES", _int_arg(None, "XCOMET_RETRIES", 7))
    resolved_la_fail = _float_arg(la_fail_score, "DUALCOMET_LA_FAIL_SCORE", 0.0)

    if solution_strs is not None:
        refs = ground_truths or [""] * len(solution_strs)
        extras = extra_infos or [{} for _ in solution_strs]
        return _score_batch(
            [str(x or "") for x in solution_strs],
            [str(x or "") for x in refs],
            [dict(x or {}) for x in extras],
            xcomet_urls=resolved_xcomet,
            cometkiwi_urls=resolved_cometkiwi,
            timeout_s=resolved_timeout_s,
            retries=resolved_retries,
            la_fail_score=resolved_la_fail,
        )

    return _score_batch(
        [solution_str or ""],
        [ground_truth or ""],
        [extra_info or {}],
        xcomet_urls=resolved_xcomet,
        cometkiwi_urls=resolved_cometkiwi,
        timeout_s=resolved_timeout_s,
        retries=resolved_retries,
        la_fail_score=resolved_la_fail,
    )[0]
