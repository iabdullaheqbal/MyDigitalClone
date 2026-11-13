import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_PROVIDED = {
    "fastapi",
    "gradio",
    "gradio-client",
    "huggingface-hub",
    "pydantic",
    "pydantic-core",
    "spaces",
    "starlette",
    "torch",
    "uvicorn",
    "websockets",
}
SPACE_FORBIDDEN_PINS = SPACE_PROVIDED - {"torch"}


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_space_requirements_pin_only_project_direct_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    dependencies = {
        _normalize_name(re.match(r"([A-Za-z0-9._-]+)", spec).group(1))
        for spec in project["project"]["dependencies"]
    }
    expected_names = dependencies - SPACE_PROVIDED
    locked_versions = {
        _normalize_name(package["name"]): package["version"]
        for package in lock["package"]
        if _normalize_name(package["name"]) in expected_names
    }
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = requirements.splitlines()
    assert lines[:2] == [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "torch",
    ]
    assert lines.count("torch") == 1
    actual_versions = {}
    for line in lines[2:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"\s*([A-Za-z0-9._-]+)==([^\s;]+)\s*", line)
        assert match, f"Expected a direct dependency pin, got: {line}"
        actual_versions[_normalize_name(match.group(1))] = match.group(2)

    assert actual_versions == {
        name: locked_versions[name] for name in expected_names
    }
    assert not (actual_versions.keys() & SPACE_FORBIDDEN_PINS)


def test_space_uses_gradio_sdk_with_python_312() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "sdk: gradio" in readme
    assert 'sdk_version: "6.31.0"' in readme
    assert 'python_version: "3.12.12"' in readme
    assert "app_file: app.py" in readme
    assert not (ROOT / "Dockerfile").exists()
