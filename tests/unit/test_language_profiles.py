from __future__ import annotations

from accel.language_profiles import (
    resolve_enabled_verify_groups,
    resolve_extension_language_map,
    resolve_extension_verify_group_map,
    resolve_selected_language_profiles,
)


def test_language_profiles_use_builtin_defaults() -> None:
    cfg = {}
    selected = resolve_selected_language_profiles(cfg)
    assert selected == ["python", "typescript"]
    ext_map = resolve_extension_language_map(cfg)
    assert ext_map[".py"] == "python"
    assert ext_map[".ts"] == "typescript"
    assert ext_map[".tsx"] == "typescript"


def test_language_profiles_allow_custom_registry() -> None:
    cfg = {
        "language_profiles": ["go"],
        "language_profile_registry": {
            "go": {
                "extensions": [".go"],
                "verify_group": "go",
            }
        },
    }
    selected = resolve_selected_language_profiles(cfg)
    assert selected == ["go"]
    ext_map = resolve_extension_language_map(cfg)
    assert ext_map == {".go": "go"}
    verify_groups = resolve_enabled_verify_groups(cfg)
    assert verify_groups == ["go"]
    verify_ext_map = resolve_extension_verify_group_map(cfg)
    assert verify_ext_map == {".go": "go"}
