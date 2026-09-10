"""Emit the C beam-search oracle fixtures the GPU port is graded against."""

from pathlib import Path

import numpy as np

from TetrisEnv.CB2BSearch import CB2BSearch

# Board geometry
ROWS, COLS = 40, 10
FULL = 0x3FF
BOT = ROWS - 1

# Search call shape
MAX_ROOTS = 1024
MAX_LEN = 15
QUEUE_LEN = 10
GARBAGE_PUSH_DELAY = 1
BAG_SEEN = 0

# tag, search_depth, beam_width, queue_len
CONFIGS = (
    ("d1w32", 1, 32, 10),
    ("d2w128", 2, 128, 10),
    ("d2w32", 2, 32, 10),
    ("d4w32", 4, 32, 10),
    ("d4w32q2", 4, 32, 2),
    ("d16w64q5", 16, 64, 5),
)

# Position grid
SEED = 20260907
RANDOM_BOARDS = 18
VARIANTS_PER_BOARD = 6
CROSS_STRIDE = 5
HOLDS = (0, 1, 6)
B2B_COMBO = ((-1, -1), (3, 0), (12, 2))
GARBAGE = (0, 2, 6)

MAX_LOCKS = 20000

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "teacher_beam.npz"

_COL_BITS = (np.uint16(1) << np.arange(COLS, dtype=np.uint16)).astype(np.uint16)


def masks_of(grid):
    """Row bitmasks of a float32 occupancy grid, bit c set when column c is filled."""
    occupied = (np.asarray(grid) != 0).astype(np.uint16)
    return (occupied * _COL_BITS).sum(axis=1, dtype=np.uint16)


def grid_of(masks):
    """Float32 occupancy grid of row bitmasks, the inverse of masks_of."""
    rows = np.asarray(masks, dtype=np.uint16)[..., None]
    return ((rows & _COL_BITS) > 0).astype(np.float32)


def decode_action(index):
    """(is_hold, rot, norm_col, spin) of a packed action index."""
    # b2b_search.c:2411
    index = int(index)
    return index // 160, (index % 160) // 40, (index % 40) // 4, index % 4


def board_of(spec):
    """Full-height board grid from a {row: bitmask} archetype spec."""
    masks = np.zeros(ROWS, dtype=np.uint16)
    for row, mask in spec.items():
        masks[row] = np.uint16(mask)
    grid = grid_of(masks)
    assert grid.shape == (ROWS, COLS)
    return grid


def random_spec(rng):
    """Stack of near-full rows with random gaps, plus a few loose cells above it."""
    nrows = int(rng.integers(1, 17))
    spec = {}
    for j in range(nrows):
        gaps = rng.choice(COLS, size=int(rng.integers(1, 4)), replace=False)
        spec[BOT - j] = FULL & ~sum(1 << int(g) for g in gaps)
    for _ in range(int(rng.integers(0, 6))):
        r = BOT - nrows - int(rng.integers(0, 3))
        if 0 <= r < ROWS:
            spec[r] = spec.get(r, 0) | (1 << int(rng.integers(0, COLS)))
    return spec


def board_specs(rng):
    """Named board archetypes as (name, {row: bitmask}) pairs, row 0 at the top."""
    specs = [
        ("empty", {}),
        ("straw", {BOT - j: 0x1FF for j in range(5)}),
        ("fuel", {BOT: 0x27F, **{BOT - 1 - j: 0x1FF for j in range(12)}}),
        ("lone", {BOT: 0x1FF}),
        ("covered", {BOT: 0x3EF, BOT - 1: 0x010}),
        ("spin_slot", {BOT - 3: 0x018, BOT - 2: 0x018, BOT - 1: 0x018, BOT: 0x3F3}),
        ("spin_tsd", {BOT: FULL & ~0x018, BOT - 1: FULL & ~0x038, BOT - 2: 0x00F}),
        (
            "spin_zigzag",
            {BOT: FULL & ~0x300, BOT - 1: FULL & ~0x200, BOT - 2: FULL & ~0x300},
        ),
    ]
    # Tall stacks clear of the rows 17-18 spawn envelope
    for h in (16, 18, 19, 20, 21):
        specs.append((f"neardeath{h}", {BOT - j: 0x1FF for j in range(h)}))
    # Blocking the spawn box: row 17 cols 3-5 (0x38) or row 18 cols 3-6 (0x78)
    specs.append(("spawnblock_full", {BOT - j: 0x1FF for j in range(23)}))
    specs.append(
        ("spawnblock_row17", {17: 0x008, **{BOT - j: 0x1FF for j in range(4)}})
    )
    specs.append(
        ("spawnblock_row18", {18: 0x040, **{BOT - j: 0x1FF for j in range(4)}})
    )
    specs.append(("spawnblock_col4", {BOT - j: 0x010 for j in range(23)}))
    # One tall column: spawn stays clear, so DEATH_HEIGHT_CAP (35) alone decides
    for col in (0, 9):
        for h in (30, 34, 35, 36):
            specs.append((f"tallcol{col}_{h}", {BOT - j: 1 << col for j in range(h)}))
    for gap in (0x00F, 0x0F0, 0x3C0):
        specs.append((f"pc_{gap:03x}", {BOT: FULL & ~gap}))
    specs.append(("pc_two_rows", {BOT: FULL & ~0x003, BOT - 1: FULL & ~0x003}))
    specs.append(("pc_tetris_well", {BOT - j: FULL & ~0x200 for j in range(4)}))
    for col in (0, 4, 9):
        specs.append(
            (f"triple_col{col}", {BOT - j: FULL & ~(1 << col) for j in range(3)})
        )
    for d in (4, 8, 12):
        specs.append((f"well{d}", {BOT - j: FULL & ~0x200 for j in range(d)}))
    for i in range(RANDOM_BOARDS):
        specs.append((f"rand{i}", random_spec(rng)))
    return specs


def build_positions():
    """Board archetypes crossed with hold, b2b/combo, garbage and a random queue."""
    rng = np.random.default_rng(SEED)
    specs = board_specs(rng)
    cross = [
        (hold, b2b, combo, garbage)
        for hold in HOLDS
        for b2b, combo in B2B_COMBO
        for garbage in GARBAGE
    ]
    grids = []
    queues = []
    pos = {k: [] for k in ("name", "active", "hold", "b2b", "combo", "garbage")}
    for bi, (name, spec) in enumerate(specs):
        grid = board_of(spec)
        for j in range(VARIANTS_PER_BOARD):
            hold, b2b, combo, garbage = cross[(bi + CROSS_STRIDE * j) % len(cross)]
            grids.append(grid)
            queues.append(rng.integers(1, 8, size=QUEUE_LEN, dtype=np.int32))
            pos["name"].append(name)
            pos["active"].append(1 + (bi * VARIANTS_PER_BOARD + j) % 7)
            pos["hold"].append(hold)
            pos["b2b"].append(b2b)
            pos["combo"].append(combo)
            pos["garbage"].append(garbage)
    return grids, queues, pos


def run_config(search, grids, queues, pos, depth, width, qlen):
    """Per-root candidates for one search config, padded to MAX_ROOTS."""
    p = len(grids)
    best = np.full(p, -1, np.int32)
    counts = np.zeros(p, np.int32)
    acts = np.full((p, MAX_ROOTS), -1, np.int32)
    scores = np.zeros((p, MAX_ROOTS), np.float32)
    rows = np.full((p, MAX_ROOTS), -1, np.int32)
    values = np.zeros(p, np.float32)
    for i in range(p):
        action, _seq, cand, cand_scores, _cand_seq, landing, value = (
            search.search_with_scores(
                grids[i],
                int(pos["active"][i]),
                int(pos["hold"][i]),
                queues[i][:qlen],
                int(pos["b2b"][i]),
                int(pos["combo"][i]),
                int(pos["garbage"][i]),
                garbage_push_delay=GARBAGE_PUSH_DELAY,
                bag_seen=BAG_SEEN,
                search_depth=depth,
                beam_width=width,
                max_len=MAX_LEN,
                max_roots=MAX_ROOTS,
            )
        )
        n = len(cand)
        best[i] = action
        counts[i] = n
        acts[i, :n] = cand
        scores[i, :n] = cand_scores
        rows[i, :n] = landing
        values[i] = value
    return best, counts, acts, scores, rows, values


def run_locks(search, grids, queues, pos, counts, acts, rows):
    """lock_score over a strided sample of the roots, with the post-lock boards."""
    pairs = [(i, k) for i in range(len(grids)) for k in range(int(counts[i]))]
    stride = max(1, -(-len(pairs) // MAX_LOCKS))
    keys = (
        "pos",
        "action",
        "piece",
        "rot",
        "col",
        "row",
        "spin",
        "clears",
        "attack",
        "new_b2b",
        "new_combo",
    )
    lock = {k: [] for k in keys}
    boards = []
    for i, k in pairs[::stride]:
        action = int(acts[i, k])
        is_hold, rot, norm_col, spin = decode_action(action)
        hold = int(pos["hold"][i])
        # b2b_search.c:2858
        if not is_hold:
            piece = int(pos["active"][i])
        elif hold != 0:
            piece = hold
        else:
            piece = int(queues[i][0])
        landing = int(rows[i, k])
        new_grid, clears, attack, new_b2b, new_combo = search.lock_score(
            grids[i],
            piece,
            rot,
            norm_col,
            landing,
            spin,
            int(pos["b2b"][i]),
            int(pos["combo"][i]),
        )
        boards.append(masks_of(new_grid))
        for key, value in zip(
            keys,
            (
                i,
                action,
                piece,
                rot,
                norm_col,
                landing,
                spin,
                clears,
                attack,
                new_b2b,
                new_combo,
            ),
        ):
            lock[key].append(value)
    out = {f"lock_{k}": np.asarray(lock[k], np.int32) for k in keys if k != "attack"}
    out["lock_attack"] = np.asarray(lock["attack"], np.float32)
    out["lock_board"] = np.stack(boards) if boards else np.zeros((0, ROWS), np.uint16)
    out["lock_stride"] = np.int32(stride)
    return out


def main():
    search = CB2BSearch()
    grids, queues, pos = build_positions()
    p = len(grids)
    out = {
        "boards": np.stack([masks_of(g) for g in grids]),
        "active": np.asarray(pos["active"], np.int32),
        "hold": np.asarray(pos["hold"], np.int32),
        "b2b": np.asarray(pos["b2b"], np.int32),
        "combo": np.asarray(pos["combo"], np.int32),
        "garbage": np.asarray(pos["garbage"], np.int32),
        "queues": np.stack(queues).astype(np.int32),
        "names": np.asarray(pos["name"]),
        "config_tags": np.asarray([c[0] for c in CONFIGS]),
        "config_params": np.asarray([c[1:] for c in CONFIGS], np.int32),
        "max_roots": np.int32(MAX_ROOTS),
        "max_len": np.int32(MAX_LEN),
        "garbage_push_delay": np.int32(GARBAGE_PUSH_DELAY),
        "bag_seen": np.int32(BAG_SEEN),
    }

    d1 = None
    for tag, depth, width, qlen in CONFIGS:
        best, counts, acts, scores, rows, values = run_config(
            search, grids, queues, pos, depth, width, qlen
        )
        out[f"best_{tag}"] = best
        out[f"n_{tag}"] = counts
        out[f"acts_{tag}"] = acts
        out[f"scores_{tag}"] = scores
        out[f"rows_{tag}"] = rows
        out[f"value_{tag}"] = values
        if tag == "d1w32":
            d1 = (counts, acts, rows)
        print(f"{tag:9s} positions {p:5d}  roots {int(counts.sum()):7d}")

    out.update(run_locks(search, grids, queues, pos, *d1))
    print(
        f"locks     records   {len(out['lock_pos']):7d}"
        f"  stride {int(out['lock_stride'])}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(OUT)


if __name__ == "__main__":
    main()
