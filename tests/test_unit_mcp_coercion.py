"""Unit tests for mcp/coercion.py"""

from __future__ import annotations

import pytest

from accel.mcp.coercion import (
    coerce_bool,
    coerce_optional_bool,
    coerce_optional_int,
    coerce_optional_float,
    coerce_sync_timeout_action,
    coerce_context_sync_timeout_action,
    coerce_events_limit,
    to_string_list,
    to_budget_override,
    BUDGET_PRESETS,
)


class TestCoerceBool:
    def test_none_returns_default(self):
        assert coerce_bool(None, True) is True
        assert coerce_bool(None, False) is False

    def test_bool_passthrough(self):
        assert coerce_bool(True) is True
        assert coerce_bool(False) is False

    def test_int_conversion(self):
        assert coerce_bool(1) is True
        assert coerce_bool(0) is False
        assert coerce_bool(42) is True

    def test_string_true_literals(self):
        assert coerce_bool("true") is True
        assert coerce_bool("True") is True
        assert coerce_bool("TRUE") is True
        assert coerce_bool("yes") is True
        assert coerce_bool("1") is True
        assert coerce_bool("on") is True

    def test_string_false_literals(self):
        assert coerce_bool("false") is False
        assert coerce_bool("False") is False
        assert coerce_bool("no") is False
        assert coerce_bool("0") is False
        assert coerce_bool("off") is False
        assert coerce_bool("") is False
        assert coerce_bool("none") is False

    def test_unknown_string_returns_default(self):
        assert coerce_bool("maybe", True) is True
        assert coerce_bool("maybe", False) is False


class TestCoerceOptionalBool:
    def test_none_returns_none(self):
        assert coerce_optional_bool(None) is None

    def test_bool_passthrough(self):
        assert coerce_optional_bool(True) is True
        assert coerce_optional_bool(False) is False

    def test_string_true_literals(self):
        assert coerce_optional_bool("true") is True
        assert coerce_optional_bool("yes") is True

    def test_string_false_literals(self):
        assert coerce_optional_bool("false") is False
        assert coerce_optional_bool("no") is False

    def test_unknown_string_returns_none(self):
        assert coerce_optional_bool("maybe") is None


class TestCoerceOptionalInt:
    def test_none_returns_none(self):
        assert coerce_optional_int(None) is None

    def test_bool_returns_none(self):
        assert coerce_optional_int(True) is None
        assert coerce_optional_int(False) is None

    def test_int_passthrough(self):
        assert coerce_optional_int(42) == 42
        assert coerce_optional_int(-5) == -5

    def test_float_truncation(self):
        assert coerce_optional_int(3.7) == 3
        assert coerce_optional_int(3.2) == 3

    def test_string_parsing(self):
        assert coerce_optional_int("42") == 42
        assert coerce_optional_int("  100  ") == 100
        assert coerce_optional_int("3.7") == 3

    def test_empty_string_returns_none(self):
        assert coerce_optional_int("") is None
        assert coerce_optional_int("   ") is None

    def test_invalid_string_returns_none(self):
        assert coerce_optional_int("abc") is None
        assert coerce_optional_int("none") is None


class TestCoerceOptionalFloat:
    def test_none_returns_none(self):
        assert coerce_optional_float(None) is None

    def test_bool_conversion(self):
        assert coerce_optional_float(True) == 1.0
        assert coerce_optional_float(False) == 0.0

    def test_number_passthrough(self):
        assert coerce_optional_float(3.14) == 3.14
        assert coerce_optional_float(42) == 42.0

    def test_string_parsing(self):
        assert coerce_optional_float("3.14") == 3.14
        assert coerce_optional_float("  42  ") == 42.0

    def test_empty_string_returns_none(self):
        assert coerce_optional_float("") is None

    def test_invalid_string_returns_none(self):
        assert coerce_optional_float("abc") is None


class TestCoerceSyncTimeoutAction:
    def test_valid_actions(self):
        assert coerce_sync_timeout_action("poll") == "poll"
        assert coerce_sync_timeout_action("cancel") == "cancel"

    def test_case_insensitive(self):
        assert coerce_sync_timeout_action("POLL") == "poll"
        assert coerce_sync_timeout_action("Cancel") == "cancel"

    def test_invalid_returns_default(self):
        assert coerce_sync_timeout_action("invalid") == "poll"
        assert coerce_sync_timeout_action("invalid", "cancel") == "cancel"

    def test_none_returns_default(self):
        assert coerce_sync_timeout_action(None) == "poll"


class TestCoerceContextSyncTimeoutAction:
    def test_valid_actions(self):
        assert coerce_context_sync_timeout_action("fallback_async") == "fallback_async"
        assert coerce_context_sync_timeout_action("cancel") == "cancel"

    def test_poll_maps_to_fallback_async(self):
        assert coerce_context_sync_timeout_action("poll") == "fallback_async"

    def test_invalid_returns_default(self):
        assert coerce_context_sync_timeout_action("invalid") == "fallback_async"

    def test_none_returns_default(self):
        assert coerce_context_sync_timeout_action(None) == "fallback_async"


class TestCoerceEventsLimit:
    def test_valid_int(self):
        assert coerce_events_limit(50) == 50

    def test_none_returns_default(self):
        assert coerce_events_limit(None) == 30
        assert coerce_events_limit(None, default_value=20) == 20

    def test_clamped_to_min(self):
        assert coerce_events_limit(0) == 1
        assert coerce_events_limit(-10) == 1

    def test_clamped_to_max(self):
        assert coerce_events_limit(1000) == 500
        assert coerce_events_limit(1000, max_value=100) == 100

    def test_string_parsing(self):
        assert coerce_events_limit("50") == 50


class TestToStringList:
    def test_none_returns_empty(self):
        assert to_string_list(None) == []

    def test_list_passthrough(self):
        assert to_string_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_list_strips_whitespace(self):
        assert to_string_list(["  a  ", "  b  "]) == ["a", "b"]

    def test_list_filters_empty(self):
        assert to_string_list(["a", "", "b", "  ", "c"]) == ["a", "b", "c"]

    def test_comma_separated(self):
        assert to_string_list("a,b,c") == ["a", "b", "c"]
        assert to_string_list("a, b, c") == ["a", "b", "c"]

    def test_newline_separated(self):
        assert to_string_list("a\nb\nc") == ["a", "b", "c"]

    def test_semicolon_separated(self):
        assert to_string_list("a;b;c") == ["a", "b", "c"]

    def test_json_array(self):
        assert to_string_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_empty_string_returns_empty(self):
        assert to_string_list("") == []
        assert to_string_list("   ") == []


class TestToBudgetOverride:
    def test_none_returns_empty(self):
        assert to_budget_override(None) == {}

    def test_preset_names(self):
        result = to_budget_override("small")
        assert result == BUDGET_PRESETS["small"]

    def test_preset_aliases(self):
        result = to_budget_override("sm")
        assert result == BUDGET_PRESETS["small"]

        result = to_budget_override("m")
        assert result == BUDGET_PRESETS["medium"]

    def test_dict_passthrough(self):
        input_dict = {"max_chars": 5000, "max_snippets": 10}
        result = to_budget_override(input_dict)
        assert result == input_dict

    def test_dict_filters_keys(self):
        input_dict = {"max_chars": 5000, "unknown_key": 123}
        result = to_budget_override(input_dict)
        assert result == {"max_chars": 5000}
        assert "unknown_key" not in result

    def test_json_object_string(self):
        result = to_budget_override('{"max_chars": 5000}')
        assert result == {"max_chars": 5000}

    def test_key_value_string(self):
        result = to_budget_override("max_chars=5000,max_snippets=10")
        assert result == {"max_chars": 5000, "max_snippets": 10}

    def test_empty_string_returns_empty(self):
        assert to_budget_override("") == {}

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="unsupported budget preset"):
            to_budget_override("invalid_preset_name")
