from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_space_requirements_work_with_hugging_face_build_dependencies() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirement_lines = requirements.splitlines()

    assert "--hash=" not in requirements
    assert not any(line.startswith("torch==") for line in requirement_lines)
    assert not any(
        line.startswith(("cuda-", "nvidia-", "triton=="))
        for line in requirement_lines
    )
