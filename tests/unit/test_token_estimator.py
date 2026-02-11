from __future__ import annotations

from accel.token_estimator import estimate_tokens_for_text, estimate_tokens_from_chars


def test_estimate_tokens_for_text_heuristic_with_calibration() -> None:
    text = "abc " * 100
    result = estimate_tokens_for_text(
        text,
        backend="heuristic",
        calibration=1.5,
        fallback_chars_per_token=4.0,
    )
    assert result["backend_used"] == "heuristic"
    assert float(result["calibration"]) == 1.5
    assert int(result["raw_tokens"]) >= 1
    assert int(result["estimated_tokens"]) >= int(result["raw_tokens"])


def test_estimate_tokens_from_chars_respects_chars_per_token() -> None:
    result = estimate_tokens_from_chars(
        8000,
        chars_per_token=5.0,
        calibration=1.2,
    )
    assert int(result["raw_tokens"]) == 1600
    assert int(result["estimated_tokens"]) == 1920
