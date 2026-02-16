from __future__ import annotations

from pathlib import Path

from accel.query.content_search import search_code_content


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_search_code_content_literal_case_insensitive(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/frontend/SettingsModal.tsx",
        "\n".join(
            [
                'const provider = "anthropic_compat";',
                'const headers = {"x-test": "1"};',
            ]
        ),
    )
    _write(
        tmp_path / "src/backend/llm.py",
        "\n".join(
            [
                'PROVIDER = "ANTHROPIC_COMPAT"',
                "DEFAULT_HEADERS = {}",
            ]
        ),
    )
    _write(
        tmp_path / "node_modules/pkg/index.js",
        'const provider = "anthropic_compat";',
    )

    result = search_code_content(
        project_dir=tmp_path,
        pattern="Anthropic_Compat",
        use_regex=False,
        context_lines=1,
        max_results=10,
    )

    assert result["status"] == "ok"
    assert result["result_count"] == 2
    assert result["files_with_matches"] == 2
    assert result["matches"][0]["file"] == "src/backend/llm.py"
    assert result["matches"][1]["file"] == "src/frontend/SettingsModal.tsx"


def test_search_code_content_regex_and_file_filter(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/backend/llm.py",
        "\n".join(
            [
                "def save_llm_config():",
                "    custom_header = get_custom_header()",
                "    return custom_header",
            ]
        ),
    )
    _write(
        tmp_path / "src/frontend/LLMSettingsTab.tsx",
        'const customHeader = "custom-header";',
    )

    result = search_code_content(
        project_dir=tmp_path,
        pattern=r"custom[_-]header",
        use_regex=True,
        file_patterns=["*.py"],
        context_lines=0,
        max_results=10,
    )

    assert result["status"] == "ok"
    assert result["result_count"] >= 2
    assert all(match["file"].endswith(".py") for match in result["matches"])


def test_search_code_content_include_exclude_and_limit(tmp_path: Path) -> None:
    _write(tmp_path / "src/frontend/StrictViewAdapter.ts", "headers = {}")
    _write(tmp_path / "src/frontend/LegacyViewAdapter.ts", "headers = {}")
    _write(tmp_path / "src/backend/llm.py", "headers = {}")

    result = search_code_content(
        project_dir=tmp_path,
        pattern="headers",
        use_regex=False,
        include_patterns=["src/frontend/**"],
        exclude_patterns=["**/Legacy*"],
        max_results=1,
    )

    assert result["status"] == "ok"
    assert result["result_count"] == 1
    assert result["truncated"] is True
    assert result["matches"][0]["file"] == "src/frontend/StrictViewAdapter.ts"


def test_search_code_content_invalid_regex(tmp_path: Path) -> None:
    _write(tmp_path / "a.py", "x = 1")

    result = search_code_content(
        project_dir=tmp_path,
        pattern="(",
        use_regex=True,
    )

    assert result["status"] == "error"
    assert result["error"] == "invalid_pattern"
