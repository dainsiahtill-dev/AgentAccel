from __future__ import annotations

import pytest

from accel.schema.contracts import (
    enforce_context_pack_contract,
    enforce_context_payload_contract,
    enforce_verify_summary_contract,
)


def test_context_pack_contract_warn_repairs_invalid_fields() -> None:
    broken = {
        "version": 0,
        "task": "",
        "top_files": "invalid",
        "snippets": [{"path": 123, "start_line": "x", "end_line": 0, "content": 777}],
        "verify_plan": {"target_tests": "bad", "target_checks": 1},
    }
    fixed, warnings, repairs = enforce_context_pack_contract(broken, mode="warn")
    assert int(fixed.get("version", 0)) == 1
    assert isinstance(fixed.get("top_files"), list)
    assert isinstance(fixed.get("snippets"), list)
    assert isinstance(fixed.get("verify_plan", {}).get("target_tests", []), list)
    assert isinstance(fixed.get("verify_plan", {}).get("target_checks", []), list)
    assert repairs >= 1
    assert warnings


def test_context_pack_contract_strict_raises() -> None:
    with pytest.raises(ValueError):
        enforce_context_pack_contract({"version": 0}, mode="strict")


def test_context_payload_contract_repairs_numbers_and_warnings_list() -> None:
    payload = {
        "estimated_tokens": "10",
        "estimated_source_tokens": None,
        "estimated_changed_files_tokens": -1,
        "estimated_snippets_only_tokens": "5",
        "selected_tests_count": "2",
        "selected_checks_count": "3",
        "warnings": "invalid",
    }
    fixed, warnings, repairs = enforce_context_payload_contract(payload, mode="warn")
    assert int(fixed.get("estimated_tokens", 0)) == 10
    assert int(fixed.get("estimated_changed_files_tokens", 0)) == 0
    assert isinstance(fixed.get("warnings"), list)
    assert repairs >= 1
    assert warnings


def test_verify_summary_contract_prefers_terminal_status() -> None:
    summary = {"latest_state": "running", "latest_stage": "running"}
    status = {"state": "cancelled", "stage": "cancelled"}
    fixed, warnings, repairs = enforce_verify_summary_contract(summary, status=status, mode="warn")
    assert fixed.get("latest_state") == "cancelled"
    assert fixed.get("state_source") == "status_terminal"
    assert int(fixed.get("constraint_repair_count", 0)) >= 1
    assert repairs >= 1
    assert warnings
