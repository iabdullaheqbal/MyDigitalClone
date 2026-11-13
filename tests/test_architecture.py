import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "clone"
REQUIRED_MODULES = (
    "config.py",
    "persona.py",
    "types.py",
    "ingestion/loader.py",
    "ingestion/chunker.py",
    "indexing/store.py",
    "indexing/builder.py",
    "retrieval/tokenizer.py",
    "retrieval/bm25.py",
    "retrieval/dense.py",
    "retrieval/fusion.py",
    "retrieval/reranker.py",
    "retrieval/hybrid.py",
    "generation/prompts.py",
    "generation/llm_client.py",
    "generation/intent.py",
    "generation/topics.py",
    "generation/responses.py",
    "generation/query_rewriter.py",
    "generation/guard.py",
    "generation/answerer.py",
    "chat/session.py",
    "guardrails/input_checks.py",
    "guardrails/output_checks.py",
    "service.py",
    "ui/layout.py",
    "ui/handlers.py",
    "ui/formatting.py",
    "evaluation/dataset.py",
    "evaluation/metrics.py",
    "evaluation/runner.py",
    "evaluation/report.py",
    "evaluation/manual_scoring.py",
    "evaluation/behavior.py",
)


def test_app_is_a_small_entrypoint() -> None:
    app = ROOT / "app.py"
    source = app.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 40
    assert not any(term in source.lower() for term in ("chunk_text", "bm25", "prompt"))
    assert "from clone import hf_runtime" in source
    assert "demo = create_app()" in source
    assert 'if __name__ == "__main__":' in source


def test_required_architecture_modules_exist() -> None:
    missing = [path for path in REQUIRED_MODULES if not (CLONE / path).is_file()]

    assert not missing, f"Missing architecture modules: {', '.join(missing)}"


def test_retrieval_does_not_import_generation() -> None:
    violations = []
    for path in (CLONE / "retrieval").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
            elif isinstance(node, ast.Import):
                imported = ",".join(alias.name for alias in node.names)
            else:
                continue
            if imported == "clone.generation" or imported.startswith("clone.generation."):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")

    assert not violations, "; ".join(violations)
