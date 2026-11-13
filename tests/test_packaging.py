from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_space_requirements_use_space_torch_without_hashes() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirement_lines = requirements.splitlines()

    assert "--hash=" not in requirements
    assert not any(line.startswith("torch") for line in requirement_lines)
    assert not any(
        line.startswith(("cuda-", "nvidia-", "triton=="))
        for line in requirement_lines
    )


def test_space_uses_gradio_sdk_with_python_311() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "sdk: gradio" in readme
    assert 'sdk_version: "6.31.0"' in readme
    assert 'python_version: "3.11.9"' in readme
    assert "app_file: app.py" in readme
    assert not (ROOT / "Dockerfile").exists()
