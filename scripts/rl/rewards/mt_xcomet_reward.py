"""Sequence-level machine translation reward based on xCOMET for verl GRPO training.

The custom reward entry point is ``compute_score``. This module normalizes text,
calls the scoring service, applies the configured score transform, and clips the
result to the requested range.

The xCOMET model is kept in a persistent HTTP service so reward workers can
reuse one loaded model instead of loading a copy in every worker.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from itertools import count
from typing import Any

import requests


# Module logger for request failures, retries, and diagnostics.
LOGGER = logging.getLogger("mt_xcomet_reward")

# Set XCOMET_DEBUG_CONN=1 to enable urllib3 connection-pool diagnostics.
if os.getenv("XCOMET_DEBUG_CONN", "").strip().lower() in ("1", "true", "yes", "on"):
    logging.getLogger("urllib3.connectionpool").setLevel(logging.DEBUG)
    # Add a stderr handler when the root logger has no configured handlers.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    LOGGER.warning("XCOMET_DEBUG_CONN is enabled; urllib3 connection-pool diagnostics are active.")

# EOS markers that may remain in generated output.
EOS_MARKERS = ("<eos>", "</s>")

# Default xCOMET endpoint used when no URL is configured explicitly.
DEFAULT_XCOMET_URL = "http://127.0.0.1:8008/score"

# Round-robin counter for selecting among multiple xCOMET endpoints.
# Starting from the process ID spreads the initial selection across workers.
_URL_COUNTER = count(os.getpid())

# Keep one requests.Session per reward worker thread.
_THREAD_LOCAL = threading.local()


def _strip_eos(text: str) -> str:
    """Truncate at the first EOS marker while preserving surrounding whitespace."""
    text = text or ""
    # Do not strip leading or trailing whitespace.
    for marker in EOS_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def _normalize(text: str) -> str:
    """Normalize text for xCOMET while removing only EOS markers."""
    text = _strip_eos(text)
    # text = text.replace("\u3000", " ")   # fullwidth space (U+3000) -> half-width space
    # text = re.sub(r"\s+", " ", text)     # collapse consecutive whitespace into one space
    # return text.strip()                  # strip leading/trailing whitespace
    return text


def _extract_source(extra_info: dict[str, Any]) -> str:
    """Extract the source sentence from ``extra_info`` for xCOMET scoring.

    Lookup order:
      1. ``source``, ``src``, or ``source_text`` in ``extra_info``.
      2. A source-language-prefixed line in the prompt.
      3. Raise an error when no source can be found.
    """
    for key in ("source", "src", "source_text"):
        value = extra_info.get(key)
        if value:
            return str(value)

    prompt = str(extra_info.get("prompt") or "")
    source_language = str(extra_info.get("source_language") or "")
    if source_language:
        prefix = f"{source_language}:"
        for line in prompt.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()

    raise KeyError("xCOMET reward requires extra_info['source'] or extra_info['src'].")


def _float_arg(value: Any, env_name: str, default: float) -> float:
    """Resolve a floating-point option from an argument, environment, or default."""
    if value is not None:
        return float(value)
    return float(os.getenv(env_name, default))


def _int_arg(value: Any, env_name: str, default: int) -> int:
    """Resolve an integer option from an argument, environment, or default."""
    if value is not None:
        return int(value)
    return int(os.getenv(env_name, default))


def _bool_arg(value: Any, env_name: str, default: bool) -> bool:
    """Resolve a boolean option from an argument, environment, or default.

    The values 1/true/yes/on are treated as true, case-insensitively.
    Other values are treated as false.
    """
    if value is not None:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_urls(xcomet_urls: Any = None, xcomet_url: Any = None) -> list[str]:
    """Resolve a list of xCOMET endpoints, optionally supplied as CSV text.

    Precedence is ``xcomet_urls`` argument, ``XCOMET_URLS`` environment
    variable, ``xcomet_url`` argument, ``XCOMET_URL`` environment variable,
    then the built-in default.
    """
    raw_value = xcomet_urls or os.getenv("XCOMET_URLS") or xcomet_url or os.getenv("XCOMET_URL") or DEFAULT_XCOMET_URL
    if isinstance(raw_value, (list, tuple)):
        urls = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        urls = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not urls:
        raise ValueError("At least one xCOMET URL is required.")
    return urls


def _select_url(xcomet_urls: list[str], skip: set[str] | None = None) -> str:
    """Select an endpoint in round-robin order, skipping failed endpoints when possible."""
    candidates = [url for url in xcomet_urls if not skip or url not in skip]
    if not candidates:
        candidates = xcomet_urls
    return candidates[next(_URL_COUNTER) % len(candidates)]


def _get_session() -> requests.Session:
    """Get a lazily initialized, thread-local requests.Session.

    Set ``XCOMET_NO_KEEPALIVE=1`` to send ``Connection: close`` and avoid
    stale keep-alive connections.
    """
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        # Ignore proxy settings inherited from the environment.
        session.trust_env = False
        if os.getenv("XCOMET_NO_KEEPALIVE", "").strip().lower() in ("1", "true", "yes", "on"):
            # Disable connection reuse for this session.
            session.headers["Connection"] = "close"
        _THREAD_LOCAL.session = session
    return session


def _xcomet_post(items: list[dict[str, str]], xcomet_urls: list[str], timeout_s: float, retries: int) -> list[dict[str, Any]]:
    """Submit a batch to xCOMET with retries and endpoint failover.

    Each item contains ``src`` and ``mt``, and may contain ``ref``. An empty
    batch returns immediately. The response must contain one result per item.

    ``retries >= 0`` allows ``retries + 1`` attempts. ``retries < 0`` enables
    infinite retrying until an endpoint succeeds.
    """
    if not items:
        return []

    payload = {"items": items}
    last_error: Exception | None = None
    session = _get_session()
    infinite = retries < 0
    max_attempts = None if infinite else retries + 1
    failed_urls: set[str] = set()
    attempt = 0
    while infinite or attempt < max_attempts:
        # In infinite mode, retry all endpoints after completing a failed round.
        if infinite and failed_urls and len(failed_urls) >= len(xcomet_urls):
            failed_urls = set()
        xcomet_url = _select_url(xcomet_urls, skip=failed_urls)
        try:
            response = session.post(xcomet_url, json=payload, timeout=timeout_s)
            response.raise_for_status()
            data = response.json()
            results = data.get("results")
            if not isinstance(results, list) or len(results) != len(items):
                raise RuntimeError(f"xCOMET service returned malformed results: {data}")
            return results
        except Exception as exc:
            # Mark the endpoint as failed and retry after a bounded backoff.
            last_error = exc
            failed_urls.add(xcomet_url)
            attempt_str = f"{attempt + 1}" if infinite else f"{attempt + 1}/{max_attempts}"
            LOGGER.warning(
                "xCOMET request to %s failed (attempt %s, %d item(s)): %r",
                xcomet_url,
                attempt_str,
                len(items),
                exc,
            )
            # Cap the backoff at two seconds and keep the exponent bounded.
            if infinite or attempt < max_attempts - 1:
                time.sleep(min(2.0, 0.25 * (2 ** min(attempt, 8))))
            attempt += 1

    # Only finite retry mode reaches this point.
    LOGGER.error(
        "xCOMET request failed after %d attempt(s) for %d item(s); tried URL(s): %s",
        max_attempts,
        len(items),
        ", ".join(sorted(failed_urls)) or "<none>",
        exc_info=last_error,
    )
    raise RuntimeError(
        f"xCOMET request failed after {max_attempts} attempt(s). "
        f"Start the service first, e.g. `bash scripts/rl/servers/run_qe_server.sh`, "
        f"or set XCOMET_URLS for multiple servers. "
        f"Last error: {last_error}"
    )


def _empty_score(empty_score: float) -> dict[str, float]:
    """Build the fallback result for an empty or unscored translation."""
    return {
        "score": float(empty_score),
        "xcomet": 0.0,
        "xcomet_scaled": 0.0,
        "empty": 1.0,
        "xcomet_error_count": 0.0,
    }


def _score_batch(
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    *,
    xcomet_urls: list[str],
    timeout_s: float,
    retries: int,
    score_scale: float,
    score_shift: float,
    score_min: float,
    score_max: float,
    empty_score: float,
    use_reference: bool,
) -> list[dict[str, float]]:
    """Score a batch of translations and return one metrics dictionary per input.

    The process has three stages: normalize text, submit one batch to xCOMET,
    then apply the score transform and restore the original order. The reward
    is pure xCOMET scoring with no additional rule penalties.

    ``use_reference=True`` includes ``ref`` for reference-based scoring;
    ``False`` omits it and uses the reference-free QE path.
    """
    # Reserve output slots so results can be restored to input order.
    scores: list[dict[str, float] | None] = [None] * len(solution_strs)
    request_items: list[dict[str, str]] = []
    request_meta: list[dict[str, float]] = []
    request_indices: list[int] = []

    # Stage one: normalize each item without changing its scoring semantics.
    for idx, (solution, ground_truth, extra_info) in enumerate(zip(solution_strs, ground_truths, extra_infos, strict=True)):
        extra_info = extra_info or {}
        reference = _normalize(ground_truth)
        hypothesis = _normalize(solution)
        item = {"src": _extract_source(extra_info), "mt": hypothesis}
        if use_reference:
            item["ref"] = reference
        request_items.append(item)
        request_meta.append({"empty": 1.0 if not hypothesis else 0.0})
        request_indices.append(idx)

    # Stage two: submit the batch to xCOMET.
    xcomet_results = _xcomet_post(request_items, xcomet_urls=xcomet_urls, timeout_s=timeout_s, retries=retries)

    # Stage three: transform scores and restore the original order.
    for idx, result, meta in zip(request_indices, xcomet_results, request_meta, strict=True):
        raw_xcomet = float(result.get("score", 0.0))
        xcomet_scaled = raw_xcomet * score_scale + score_shift
        score = min(score_max, max(score_min, xcomet_scaled))
        error_spans = result.get("error_spans") or []
        scores[idx] = {
            "score": float(score),
            "xcomet": raw_xcomet,
            "xcomet_scaled": float(xcomet_scaled),
            "empty": meta["empty"],
            "xcomet_error_count": float(len(error_spans) if isinstance(error_spans, list) else 0),
        }

    # Guard against an unexpected missing result.
    return [score if score is not None else _empty_score(empty_score) for score in scores]


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
    timeout_s: float | None = None,
    retries: int | None = None,
    score_scale: float | None = None,
    score_shift: float | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    empty_score: float | None = None,
    use_reference: bool | str | None = None,
    **_: Any,
) -> dict[str, float] | list[dict[str, float]]:
    """verl custom reward entry point backed by an xCOMET HTTP service.

    The final reward is the transformed and clipped xCOMET score without
    language, repetition, or length penalties.

    Both batch calls (``solution_strs`` and related lists) and single-item calls
    (``solution_str`` and related fields) are supported. Scoring options can be
    passed explicitly, read from their ``XCOMET_*`` environment variables, or
    taken from the defaults below. Extra keyword arguments are ignored for
    compatibility with future verl versions.
    """

    # Resolve configuration from arguments, environment variables, and defaults.
    resolved_urls = _parse_urls(xcomet_urls=xcomet_urls, xcomet_url=xcomet_url)
    resolved_timeout_s = _float_arg(timeout_s, "XCOMET_TIMEOUT_S", 600.0)
    resolved_retries = _int_arg(retries, "XCOMET_RETRIES", 1)
    resolved_score_scale = _float_arg(score_scale, "XCOMET_SCORE_SCALE", 1.0)
    resolved_score_shift = _float_arg(score_shift, "XCOMET_SCORE_SHIFT", 0.0)
    resolved_score_min = _float_arg(score_min, "XCOMET_SCORE_MIN", -1.0)
    resolved_score_max = _float_arg(score_max, "XCOMET_SCORE_MAX", 1.0)
    resolved_empty_score = _float_arg(empty_score, "XCOMET_EMPTY_SCORE", -0.3)
    resolved_use_reference = _bool_arg(use_reference, "XCOMET_USE_REFERENCE", True)

    if solution_strs is not None:
        # Fill missing references and metadata to the batch length.
        refs = ground_truths or [""] * len(solution_strs)
        extras = extra_infos or [{} for _ in solution_strs]
        return _score_batch(
            [str(item or "") for item in solution_strs],
            [str(item or "") for item in refs],
            [dict(item or {}) for item in extras],
            xcomet_urls=resolved_urls,
            timeout_s=resolved_timeout_s,
            retries=resolved_retries,
            score_scale=resolved_score_scale,
            score_shift=resolved_score_shift,
            score_min=resolved_score_min,
            score_max=resolved_score_max,
            empty_score=resolved_empty_score,
            use_reference=resolved_use_reference,
        )

    # Reuse the batch path for a single item and return its first result.
    return _score_batch(
        [solution_str or ""],
        [ground_truth or ""],
        [extra_info or {}],
        xcomet_urls=resolved_urls,
        timeout_s=resolved_timeout_s,
        retries=resolved_retries,
        score_scale=resolved_score_scale,
        score_shift=resolved_score_shift,
        score_min=resolved_score_min,
        score_max=resolved_score_max,
        empty_score=resolved_empty_score,
        use_reference=resolved_use_reference,
    )[0]
