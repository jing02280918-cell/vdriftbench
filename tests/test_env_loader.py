import os

import pytest

from vdriftbench.env_loader import load_dotenv_if_present


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("DEEPSEEK_API_KEY", "VDB_TARGET_API_KEY", "SOME_QUOTED_VAR", "ALREADY_SET"):
        monkeypatch.delenv(key, raising=False)


def test_loads_simple_key_value_pairs(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-abc123\nVDB_TARGET_API_KEY=sk-def456\n")

    loaded = load_dotenv_if_present(str(env_file))

    assert set(loaded) == {"DEEPSEEK_API_KEY", "VDB_TARGET_API_KEY"}
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-abc123"
    assert os.environ["VDB_TARGET_API_KEY"] == "sk-def456"


def test_ignores_comments_and_blank_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\n\nDEEPSEEK_API_KEY=sk-abc123\n\n# trailing comment\n")

    loaded = load_dotenv_if_present(str(env_file))

    assert loaded == ["DEEPSEEK_API_KEY"]
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-abc123"


def test_strips_quotes_and_export_prefix(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('export SOME_QUOTED_VAR="sk-quoted-value"\n')

    load_dotenv_if_present(str(env_file))

    assert os.environ["SOME_QUOTED_VAR"] == "sk-quoted-value"


def test_does_not_override_existing_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("ALREADY_SET=from-file\n")

    loaded = load_dotenv_if_present(str(env_file))

    assert loaded == []
    assert os.environ["ALREADY_SET"] == "from-shell"


def test_missing_file_returns_empty_list(tmp_path):
    missing = tmp_path / "does_not_exist.env"

    assert load_dotenv_if_present(str(missing)) == []


def test_falls_back_to_first_existing_candidate(tmp_path):
    missing = tmp_path / "missing.env"
    present = tmp_path / "present.env"
    present.write_text("DEEPSEEK_API_KEY=sk-fallback\n")

    loaded = load_dotenv_if_present(str(missing), str(present))

    assert loaded == ["DEEPSEEK_API_KEY"]
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-fallback"
