from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_space_requirements_pin_cpu_torch_without_hashes() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirement_lines = requirements.splitlines()

    assert "--hash=" not in requirements
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.15.0+cpu ; sys_platform == 'linux'" in requirements
    assert not any(
        line.startswith(("cuda-", "nvidia-", "triton=="))
        for line in requirement_lines
    )


def test_space_uses_python_311_docker_runtime() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert "FROM python:3.11-slim" in dockerfile
