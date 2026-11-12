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


def test_from_env_rejects_invalid_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_WORDS", "0")

    with pytest.raises(ValueError, match="CHUNK_WORDS"):
        Settings.from_env()


def test_from_env_normalizes_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_BASE_URL", "https://example.test/v1///")

    assert Settings.from_env().groq_base_url == "https://example.test/v1"
