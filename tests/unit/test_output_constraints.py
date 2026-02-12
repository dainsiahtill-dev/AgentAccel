from __future__ import annotations

import pytest

from accel.schema.contracts import (
    enforce_context_pack_contract,
    enforce_context_payload_contract,
    enforce_verify_events_payload_contract,
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


def test_context_payload_contract_strict_allows_missing_warnings_field() -> None:
    payload = {
        "estimated_tokens": 10,
        "estimated_source_tokens": 20,
        "estimated_changed_files_tokens": 5,
        "estimated_snippets_only_tokens": 8,
        "selected_tests_count": 1,
        "selected_checks_count": 2,
    }
    fixed, warnings, repairs = enforce_context_payload_contract(payload, mode="strict")
    assert repairs == 0
    assert warnings == []
    assert fixed.get("warnings") == []


def test_verify_summary_contract_prefers_terminal_status() -> None:
    summary = {"latest_state": "running", "latest_stage": "running"}
    status = {"state": "cancelled", "stage": "cancelled"}
    fixed, warnings, repairs = enforce_verify_summary_contract(summary, status=status, mode="warn")
    assert fixed.get("latest_state") == "cancelled"
    assert fixed.get("state_source") == "status_terminal"
    assert int(fixed.get("constraint_repair_count", 0)) >= 1
    assert repairs >= 1
    assert warnings


def test_verify_events_payload_contract_warn_repairs_invalid_payload() -> None:
    broken = {
        "job_id": 123,
        "events": [{"event": "", "seq": 0}, "bad"],
        "count": 999,
        "total_available": -1,
        "truncated": "yes",
        "max_events": 0,
        "since_seq": -5,
        "summary": {"constraint_repair_count": -2, "event_type_counts": []},
    }
    fixed, warnings, repairs = enforce_verify_events_payload_contract(broken, mode="warn")
    assert isinstance(fixed.get("job_id"), str)
    assert isinstance(fixed.get("events"), list)
    assert int(fixed.get("count", 0)) == len(fixed.get("events", []))
    assert int(fixed.get("max_events", 0)) >= 1
    assert int(fixed.get("since_seq", 0)) >= 0
    assert repairs >= 1
    assert warnings


def test_verify_events_payload_contract_strict_raises() -> None:
    with pytest.raises(ValueError):
        enforce_verify_events_payload_contract(
            {
                "job_id": "verify_abc",
                "events": [{"event": ""}],
                "count": 1,
                "total_available": 1,
                "truncated": False,
                "max_events": 30,
                "since_seq": 0,
            },
            mode="strict",
        )
