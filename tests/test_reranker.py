import sys
from types import ModuleType

import pytest

from clone.retrieval.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_forces_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str) -> None:
            received["model_name"] = model_name
            received["device"] = device

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    CrossEncoderReranker("test-reranker")._load_model()

    assert received == {"model_name": "test-reranker", "device": "cpu"}
