from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_space_requirements_use_cpu_torch_without_gpu_runtime_packages() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.15.0+cpu ; sys_platform == 'linux'" in requirements
    assert not any(
        line.startswith(("cuda-", "nvidia-", "triton=="))
        for line in requirements.splitlines()
    )
