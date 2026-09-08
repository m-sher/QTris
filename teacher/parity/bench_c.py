"""Time the C beam oracle on the parity fixture positions."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "teacher_beam.npz"
)
_COL_BITS = (np.uint16(1) << np.arange(10, dtype=np.uint16)).astype(np.uint16)

# tag, depth, width, queue_len
CONFIGS = {
    "datagen": (10, 256, 10),
    "august": (10, 512, 10),
    "dagger": (16, 200, 5),
}


def grid_of(masks):
    """Float32 occupancy grid of row bitmasks."""
    rows = np.asarray(masks, np.uint16)[..., None]
    return ((rows & _COL_BITS) > 0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="datagen")
    p.add_argument("--positions", type=int, default=64)
    p.add_argument("--repeats", type=int, default=1)
    a = p.parse_args()

    from TetrisEnv.CB2BSearch import CB2BSearch

    depth, width, qlen = CONFIGS[a.config]
    d = np.load(FIXTURE)
    n = min(a.positions, len(d["boards"]))
    grids = [grid_of(d["boards"][i]) for i in range(n)]
    search = CB2BSearch()

    def one(i):
        search.search_with_scores(
            grids[i],
            int(d["active"][i]),
            int(d["hold"][i]),
            d["queues"][i][:qlen],
            int(d["b2b"][i]),
            int(d["combo"][i]),
            int(d["garbage"][i]),
            search_depth=depth,
            beam_width=width,
            max_len=15,
            max_roots=1024,
        )

    one(0)
    best = None
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        for i in range(n):
            one(i)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)

    threads = os.environ.get("OMP_NUM_THREADS", "default")
    print(
        f"C {a.config} d{depth} w{width} q{qlen} threads={threads}: "
        f"{n} positions in {best:.2f}s -> {best / n * 1e3:.1f} ms/move, "
        f"{n / best:.2f} moves/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
