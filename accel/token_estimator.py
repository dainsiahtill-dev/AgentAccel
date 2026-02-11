from __future__ import annotations

import math
from typing import Any

TokenEstimate = dict[str, Any]
_TOKEN_BACKENDS = {"auto", "tiktoken", "heuristic"}


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return float(default)
    return float(parsed)


def _normalize_backend(value: Any) -> str:
    token = str(value or "auto").strip().lower()
    return token if token in _TOKEN_BACKENDS else "auto"


def _estimate_with_tiktoken(text: str, *, model: str, encoding: str) -> tuple[int, str]:
    import tiktoken  # type: ignore[import-untyped]

    encoder = None
    if model:
        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            encoder = None
    if encoder is None:
        encoder = tiktoken.get_encoding(encoding)
    token_count = len(encoder.encode(text))
    encoder_name = str(getattr(encoder, "name", encoding))
    return int(token_count), encoder_name


def estimate_tokens_for_text(
    text: str,
    *,
    backend: Any = "auto",
    model: Any = "",
    encoding: Any = "cl100k_base",
    calibration: Any = 1.0,
    fallback_chars_per_token: Any = 4.0,
) -> TokenEstimate:
    content = str(text or "")
    backend_value = _normalize_backend(backend)
    model_value = str(model or "").strip()
    encoding_value = str(encoding or "cl100k_base").strip() or "cl100k_base"
    calibration_value = _positive_float(calibration, 1.0)
    fallback_cpt = _positive_float(fallback_chars_per_token, 4.0)

    raw_tokens = 0
    backend_used = "heuristic"
    encoding_used = f"chars/{fallback_cpt:g}"
    fallback_reason = ""

    if backend_value in {"auto", "tiktoken"}:
        try:
            raw_tokens, encoding_used = _estimate_with_tiktoken(
                content,
                model=model_value,
                encoding=encoding_value,
            )
            backend_used = "tiktoken"
        except Exception as exc:
            fallback_reason = f"tiktoken_unavailable:{exc.__class__.__name__}"
            raw_tokens = 0

    if raw_tokens <= 0:
        raw_tokens = max(1, int(math.ceil(len(content) / fallback_cpt)))
        backend_used = "heuristic"

    estimated_tokens = max(1, int(round(raw_tokens * calibration_value)))
    chars_per_token = float(len(content)) / float(raw_tokens) if raw_tokens > 0 else fallback_cpt

    return {
        "estimated_tokens": int(estimated_tokens),
        "raw_tokens": int(raw_tokens),
        "backend_requested": backend_value,
        "backend_used": backend_used,
        "encoding_requested": encoding_value,
        "encoding_used": encoding_used,
        "model": model_value,
        "calibration": float(calibration_value),
        "fallback_chars_per_token": float(fallback_cpt),
        "chars_per_token": float(chars_per_token),
        "fallback_reason": fallback_reason,
    }


def estimate_tokens_from_chars(
    chars: int,
    *,
    chars_per_token: Any,
    calibration: Any = 1.0,
) -> TokenEstimate:
    chars_value = max(0, int(chars))
    cpt = _positive_float(chars_per_token, 4.0)
    calibration_value = _positive_float(calibration, 1.0)

    raw_tokens = max(1, int(math.ceil(chars_value / cpt))) if chars_value > 0 else 1
    estimated_tokens = max(1, int(round(raw_tokens * calibration_value)))
    return {
        "estimated_tokens": int(estimated_tokens),
        "raw_tokens": int(raw_tokens),
        "chars_per_token": float(cpt),
        "calibration": float(calibration_value),
    }
