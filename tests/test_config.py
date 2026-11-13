from pathlib import Path

import pytest

from clone.config import Settings


def test_from_env_uses_safe_defaults_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.groq_api_key is None
    assert settings.groq_model == "openai/gpt-oss-20b"
    assert settings.groq_base_url == "https://api.groq.com/openai/v1"
    assert settings.data_dir == Path(__file__).resolve().parents[1] / "data"
    assert settings.strong_cosine == 0.45
    assert settings.weak_cosine == 0.20
    assert settings.show_supervisor_name is False


def test_from_env_rejects_invalid_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_WORDS", "0")

    with pytest.raises(ValueError, match="CHUNK_WORDS"):
        Settings.from_env()


def test_from_env_normalizes_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_BASE_URL", "https://example.test/v1///")

    assert Settings.from_env().groq_base_url == "https://example.test/v1"


def test_from_env_rejects_reversed_evidence_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRONG_COSINE", "0.1")
    monkeypatch.setenv("WEAK_COSINE", "0.2")

    with pytest.raises(ValueError, match="STRONG_COSINE"):
        Settings.from_env()


def test_from_env_parses_supervisor_name_privacy_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOW_SUPERVISOR_NAME", "true")

    assert Settings.from_env().show_supervisor_name is True
