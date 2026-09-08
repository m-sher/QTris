"""Shared loader and GPU guard for the teacher parity gates."""

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "teacher_beam.npz"


def load():
    """The C oracle fixtures, or skip when they have not been generated."""
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} missing; run python -m teacher.parity.make_fixtures")
    return np.load(FIXTURE)


def no_hold_roots(data, tag="d1w32"):
    """(position, root) pairs whose root action places the active piece."""
    n = data[f"n_{tag}"]
    acts = data[f"acts_{tag}"]
    return [
        (i, k) for i in range(len(n)) for k in range(int(n[i])) if int(acts[i, k]) < 160
    ]
