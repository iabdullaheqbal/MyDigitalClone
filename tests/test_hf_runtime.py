import importlib
import sys


def test_runtime_probe_imports_without_spaces_and_is_a_noop(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "spaces", None)
    runtime = importlib.reload(importlib.import_module("clone.hf_runtime"))

    assert callable(runtime.zerogpu_probe)
    assert runtime.zerogpu_probe() is None
