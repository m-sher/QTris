"""Sample the placement model's opener repertoire over the whole first bag.

For each of the 7! = 5040 orderings of the first bag, set up an empty board with
that order followed by a fixed canonical continuation (so the lookahead window is
identical across orderings and the final board is a pure function of the first-bag
order), play 7 placements with the deployed MCTS config (greedy by visit count,
hold allowed), and tally the resulting boards. Prints the most common final states.

Run from the repo root:
    uv run python scripts/sample_opener.py
"""

import argparse
import itertools
import math
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from qtris.demo.utils import load_checkpoint
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.search.placement_search import placement_step
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import PyTetrisEnv

# Model + search params mirror tetrio/triangle_integration_placement.py (the deployed bot).
PIECE_DIM = 8
DEPTH = 64
NUM_HEADS = 4
NUM_LAYERS = 4
QUEUE_SIZE = 5
MAX_LEN = 15
NUM_ROW_TIERS = 2
C_PUCT = 1.5
DEFAULT_CHECKPOINT = "checkpoints/1v1_placement_az"

BAG = [
    PieceType.I,
    PieceType.J,
    PieceType.L,
    PieceType.O,
    PieceType.S,
    PieceType.T,
    PieceType.Z,
]
CANONICAL_CONTINUATION = list(
    BAG
)  # fixed second bag fed into every ordering's lookahead
TETRIO_T_FIXED = (
    1  # pin the C engine's speculative-bag seed so orderings are comparable
)


def build_net(checkpoint):
    net = PlacementPolicyValueNet(
        batch_size=1,
        piece_dim=PIECE_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=0.0,
    )
    net(
        (
            tf.keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            tf.keras.Input(shape=(QUEUE_SIZE + 2,), dtype=tf.int64),
            tf.keras.Input(shape=(3,), dtype=tf.float32),
            tf.keras.Input(shape=(None, 18), dtype=tf.float32),
            tf.keras.Input(shape=(None,), dtype=tf.bool),
        )
    )
    load_checkpoint(net, checkpoint)
    return_scale = 1.0
    try:
        reader = tf.train.load_checkpoint(tf.train.latest_checkpoint(checkpoint))
        return_scale = float(
            reader.get_tensor("return_scale/.ATTRIBUTES/VARIABLE_VALUE")
        )
        print(f"Using trained return_scale={return_scale:.3f}", file=sys.stderr)
    except Exception:
        print("No return_scale in checkpoint; using 1.0", file=sys.stderr)
    return net, return_scale


def make_env():
    return PyTetrisEnv(
        queue_size=QUEUE_SIZE,
        max_holes=999,
        max_steps=None,
        max_len=MAX_LEN,
        pathfinding=False,
        seed=None,
        idx=0,
        garbage_chance=0,
        garbage_min=0,
        garbage_max=0,
        auto_push_garbage=False,
        auto_fill_queue=False,
        num_row_tiers=NUM_ROW_TIERS,
    )


def set_opener(env, perm):
    """Reset `env` to an empty board whose stream is `perm` then the canonical bag."""
    stream = list(perm) + CANONICAL_CONTINUATION
    env.reset()
    env._board[:] = 0.0
    env._vis_board[:] = 0
    env._scorer.reset()
    env._hold_piece = PieceType.N
    env._active_piece = env._spawn_piece(stream[0])
    env._queue = list(stream[1:])
    env._next_bag = list(CANONICAL_CONTINUATION)
    env._garbage_queue = []
    env._step_num = 0
    env._last_phi = 0.0
    env._episode_ended = False
    env._tetrio_rng._t = TETRIO_T_FIXED


def play_chunk(perms, env_pool, mcts, searcher, return_scale):
    """Play 7 placements for each ordering in the chunk; return final boards + death count.
    Reuses `env_pool` (reset per ordering) so envs aren't reconstructed every chunk."""
    envs = env_pool[: len(perms)]
    for env, perm in zip(envs, perms):
        set_opener(env, perm)
    alive = [True] * len(perms)
    for _ in range(7):
        results = mcts.search(envs, return_scale, 0.0)
        for i, res in enumerate(results):
            if not alive[i]:
                continue
            if res["dead"]:
                alive[i] = False
                continue
            placement_step(envs[i], searcher, res["descriptor"])
    boards, deaths = [], 0
    for env, ok in zip(envs, alive):
        if ok:
            boards.append((env._board != 0).astype(np.uint8))
        else:
            deaths += 1
    return boards, deaths


def board_lines(occ, height):
    return [
        "".join("#" if occ[r, c] else "." for c in range(10))
        for r in range(40 - height, 40)
    ]


def stack_height(occ):
    rows = np.flatnonzero(occ.any(axis=1))
    return int(40 - rows.min()) if rows.size else 1


def print_gallery(entries, total, per_row, max_rows):
    """entries: list of (rank, count, occ) sorted by count desc."""
    width = 11
    for start in range(0, len(entries), per_row):
        block = entries[start : start + per_row]
        height = min(max_rows, max(stack_height(occ) for _, _, occ in block))
        head1 = " ".join(f"#{rank}".ljust(width) for rank, _, _ in block)
        head2 = " ".join(
            f"{count} {100 * count / total:.1f}%".ljust(width) for _, count, _ in block
        )
        print(head1.rstrip())
        print(head2.rstrip())
        rendered = [board_lines(occ, height) for _, _, occ in block]
        for row in range(height):
            print(" ".join(lines[row].ljust(width) for lines in rendered).rstrip())
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--sims", type=int, default=256, help="MCTS simulations per move")
    ap.add_argument("--leaves", type=int, default=8, help="leaves per net call")
    ap.add_argument(
        "--chunk",
        type=int,
        default=84,
        help="orderings searched in parallel (divisor of 5040 avoids XLA recompiles)",
    )
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--per-row", type=int, default=6, help="boards per gallery row")
    ap.add_argument("--max-rows", type=int, default=12, help="board rows shown")
    ap.add_argument("--limit", type=int, default=0, help="cap orderings (0 = all 5040)")
    args = ap.parse_args()

    net, return_scale = build_net(args.checkpoint)
    cfg = MCTSConfig(
        num_simulations=args.sims,
        c_puct=C_PUCT,
        dirichlet_eps=0.0,
        leaves_per_round=args.leaves,
        gamma=1.0,
        w_attack=0.05,
        w_death=1.0,
        w_b2b=0.06,
    )
    mcts = PlacementMCTS(net, cfg)
    searcher = CB2BSearch()

    perms = list(itertools.permutations(BAG, 7))
    if args.limit:
        perms = perms[: args.limit]

    # Build the env pool once, before the bar, and reuse it across chunks: the per-env
    # init prints then fire once up front instead of interleaving with the progress bar.
    env_pool = [make_env() for _ in range(min(args.chunk, len(perms)))]

    tally = Counter()
    reps = {}
    deaths = 0
    with tqdm(total=len(perms), desc="openers", unit="ordering") as pbar:
        for start in range(0, len(perms), args.chunk):
            chunk = perms[start : start + args.chunk]
            boards, d = play_chunk(chunk, env_pool, mcts, searcher, return_scale)
            deaths += d
            for occ in boards:
                key = occ.tobytes()
                tally[key] += 1
                reps.setdefault(key, occ)
            pbar.update(len(chunk))

    completed = sum(tally.values())
    distinct = tally.most_common()
    top_n = max(1, math.ceil(args.top_frac * len(distinct)))
    shown = distinct[:top_n]
    coverage = sum(c for _, c in shown)

    print(f"\nOrderings: {len(perms)}  completed: {completed}  deaths: {deaths}")
    print(f"Distinct final boards: {len(distinct)}")
    print(
        f"Showing top {args.top_frac:.0%} = {top_n} states "
        f"({100 * coverage / completed:.1f}% of completed orderings):\n"
    )
    entries = [(i + 1, count, reps[key]) for i, (key, count) in enumerate(shown)]
    print_gallery(entries, completed, args.per_row, args.max_rows)


if __name__ == "__main__":
    main()
