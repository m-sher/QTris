"""Skips the teacher GPU gates without the teacher extra, a device or the fixtures."""

import importlib
from pathlib import Path

TEACHER_TESTS = (
    "test_teacher_rules.py",
    "test_teacher_enumerate.py",
    "test_teacher_evaluate.py",
    "test_teacher_beam.py",
    "test_teacher_batch.py",
    "test_teacher_oracle.py",
    "test_teacher_demo.py",
)

FIXTURE = Path(__file__).parent / "fixtures" / "teacher_beam.npz"

collect_ignore = []


def _gpu_ready():
    """True when cupy, the teacher package and a CUDA device are all available."""
    try:
        importlib.import_module("cupy")
        importlib.import_module("teacher")
        cuda = importlib.import_module("numba.cuda")
    except ImportError:
        return False
    return cuda.is_available()


if not (FIXTURE.exists() and _gpu_ready()):
    collect_ignore.extend(TEACHER_TESTS)
