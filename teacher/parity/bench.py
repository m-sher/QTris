"""Time the GPU beam teacher on the parity fixture positions."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from teacher.api import GpuTeacher

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


def launch_overhead(reps=2000):
    """Mean wall time of one trivial numba kernel launch, in microseconds."""
    import cupy as cp
    from numba import cuda

    @cuda.jit
    def _noop(x):
        i = cuda.grid(1)
        if i < x.size:
            x[i] = x[i]

    buf = cp.zeros(64, cp.int32)
    _noop[1, 64](buf)
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _noop[1, 64](buf)
    cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="datagen")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--positions", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()

    import cupy as cp
    from numba import cuda

    depth, width, qlen = CONFIGS[a.config]
    d = np.load(FIXTURE)
    n = min(a.positions, len(d["boards"]))
    sel = np.arange(n)
    boards = grid_of(d["boards"][sel])
    active, hold = d["active"][sel], d["hold"][sel]
    queues = np.ascontiguousarray(d["queues"][sel, :qlen])
    b2b, combo, garb = d["b2b"][sel], d["combo"][sel], d["garbage"][sel]

    print(f"GPU {a.config} d{depth} w{width} q{qlen}, {n} fixture positions")
    for batch in a.batch:
        try:
            teacher = GpuTeacher(
                max_batch=batch, width=width, depth=depth, queue_len=qlen
            )
        except Exception as exc:  # allocation refused at this shape
            print(f"  B={batch:3d}  unavailable: {type(exc).__name__}: {exc}")
            continue
        chunks = [
            slice(s, min(s + batch, n)) for s in range(0, n - n % batch or n, batch)
        ]
        chunks = [c for c in chunks if c.stop - c.start == batch]
        if not chunks:
            print(f"  B={batch:3d}  skipped: fewer than {batch} positions")
            continue

        t0 = time.perf_counter()
        teacher.search_batch(
            boards[chunks[0]],
            active[chunks[0]],
            hold[chunks[0]],
            queues[chunks[0]],
            qlen,
            b2b[chunks[0]],
            combo[chunks[0]],
            garb[chunks[0]],
        )
        compile_s = time.perf_counter() - t0

        best = None
        for _ in range(a.repeats):
            cuda.synchronize()
            t0 = time.perf_counter()
            for c in chunks:
                teacher.search_batch(
                    boards[c],
                    active[c],
                    hold[c],
                    queues[c],
                    qlen,
                    b2b[c],
                    combo[c],
                    garb[c],
                )
            cuda.synchronize()
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)

        moves = len(chunks) * batch
        peak = cp.get_default_memory_pool().used_bytes() / 2**20
        print(
            f"  B={batch:3d}  {moves:4d} moves in {best:6.2f}s -> "
            f"{best / moves * 1e3:7.1f} ms/move, {moves / best:7.2f} moves/s"
            f"   first call {compile_s:5.1f}s, pool {peak:6.0f} MiB"
        )

    print(f"  numba kernel launch overhead: {launch_overhead():.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
