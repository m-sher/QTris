"""Recover placement key sequences for TETR.IO replays that only recorded hardDrop.

Uses the same C pathfinder MCTS / b2b_search enumerate with (`find_unique_placements`
+ `lock_score`). Each candidate is checked against the *next* frame's board, b2b,
combo, and piece queue. Garbage that landed after the lock is allowed as extra
bottom rows (same geometry as `_push_garbage_to_board`).

A spectate replay is one session of many rounds, so transitions that cross a round
boundary are skipped.

Writes a sibling `*.recovered.json` (same `{"mode": ..., "frames": [...]}` schema,
recovered `key_sequence` / `converted_keys` on every matched frame). Does not overwrite
the input.

    uv run python scripts/recover_spectate_keys.py [PATH]
    uv run python scripts/recover_spectate_keys.py tetrio/replays/replay_20260707_005526.json
    uv run python scripts/recover_spectate_keys.py PATH --out /tmp/out.json
"""

import argparse
import json
import os
import sys

import numpy as np

from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import PyTetrisEnv

REPLAY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tetrio", "replays"
)
DEFAULT_REPLAY = os.path.join(REPLAY_DIR, "replay_20260707_005526.json")
MAX_LEN = 15
MAX_GARBAGE_ROWS = 12

KEY_MAP = {
    1: "hold",
    2: "moveLeft",
    3: "moveRight",
    4: "dasLeft",
    5: "dasRight",
    6: "rotateCW",
    7: "rotateCCW",
    8: "rotate180",
    9: "softDrop",
    10: "hardDrop",
}


def _board40(stored):
    board = np.zeros((40, 10), dtype=np.float32)
    src = np.asarray(stored, dtype=np.float32)
    rows = min(src.shape[0], 40)
    board[-rows:] = src[-rows:]
    return board


def _occ24(board40):
    return board40[-24:] != 0


def _board_match(pred40, actual24):
    """Exact occupancy match, or pred shifted up with garbage appended at the bottom."""
    pred = _occ24(pred40)
    actual = np.asarray(actual24) != 0
    if pred.shape != actual.shape:
        return False, -1
    if np.array_equal(pred, actual):
        return True, 0
    for k in range(1, MAX_GARBAGE_ROWS + 1):
        if np.array_equal(actual[:-k], pred[k:]):
            return True, k
    return False, -1


def _expected_after(pieces, is_hold):
    active, hold, queue = int(pieces[0]), int(pieces[1]), [int(p) for p in pieces[2:]]
    if is_hold:
        if hold == PieceType.N.value:
            if not queue:
                return None
            queue = queue[1:]
        new_hold = active
    else:
        new_hold = hold
    if not queue:
        return None
    new_active = queue[0]
    new_queue = queue[1:]
    return new_active, new_hold, new_queue


def _queue_match(expected, next_pieces):
    if expected is None:
        return False
    new_active, new_hold, new_queue = expected
    nxt = [int(p) for p in next_pieces]
    if nxt[0] != new_active or nxt[1] != new_hold:
        return False
    tail = nxt[2:]
    return tail[: len(new_queue)] == new_queue


def _converted(keys):
    return [KEY_MAP[int(k)] for k in keys if int(k) not in (Keys.START, Keys.PAD)]


def _make_env():
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=999,
        max_steps=None,
        max_len=MAX_LEN,
        pathfinding=False,
        seed=None,
        idx=0,
        garbage_chance=0,
        auto_push_garbage=False,
        auto_fill_queue=False,
        use_shaping=False,
        placement_candidates=False,
        num_row_tiers=2,
    )
    env.reset()
    return env


def _load_state(env, frame):
    pieces = frame["pieces"]
    env._board = _board40(frame["board"])
    env._vis_board = env._board.copy()
    env._active_piece = env._spawn_piece(PieceType(int(pieces[0])))
    env._hold_piece = PieceType(int(pieces[1]))
    env._queue = [PieceType(int(p)) for p in pieces[2:]]
    b2b, combo, _ = frame["b2b_combo_garbage"]
    env._scorer._b2b = int(b2b)
    env._scorer._combo = int(combo)
    return int(b2b), int(combo)


def _enumerate(env, finder):
    """Same unique-placement enum MCTS uses (no-hold + hold branches)."""
    out = []
    branches = [(False, env._active_piece)]
    if env._hold_piece != PieceType.N:
        branches.append((True, env._spawn_piece(env._hold_piece)))
    elif env._queue:
        branches.append((True, env._spawn_piece(env._queue[0])))
    for is_hold, piece in branches:
        rots, ncols, lrs, spins, seqs = finder.find_unique_placements(
            board=env._board,
            piece=piece,
            max_len=MAX_LEN,
            is_hold=is_hold,
            max_out=512,
            with_sequences=True,
        )
        for i in range(len(rots)):
            out.append(
                {
                    "is_hold": is_hold,
                    "rot": int(rots[i]),
                    "norm_col": int(ncols[i]),
                    "landing_row": int(lrs[i]),
                    "spin": int(spins[i]),
                    "keys": np.asarray(seqs[i], dtype=np.int64),
                }
            )
    return out


def recover_frame(env, finder, searcher, frame, nxt):
    b2b, combo = _load_state(env, frame)
    want_b2b = int(nxt["b2b_combo_garbage"][0])
    want_combo = int(nxt["b2b_combo_garbage"][1])
    hits = []
    for cand in _enumerate(env, finder):
        expected = _expected_after(frame["pieces"], cand["is_hold"])
        if not _queue_match(expected, nxt["pieces"]):
            continue
        placed = (
            env._queue[0].value
            if cand["is_hold"] and env._hold_piece == PieceType.N
            else (
                env._hold_piece.value
                if cand["is_hold"]
                else env._active_piece.piece_type.value
            )
        )
        new_board, clears, attack, new_b2b, new_combo = searcher.lock_score(
            env._board,
            placed,
            cand["rot"],
            cand["norm_col"],
            cand["landing_row"],
            cand["spin"],
            b2b,
            combo,
        )
        if new_b2b != want_b2b or new_combo != want_combo:
            continue
        ok, garb_rows = _board_match(new_board, nxt["board"])
        if not ok:
            continue
        cand = dict(cand)
        cand.update(
            clears=int(clears),
            attack=float(attack),
            garb_rows=garb_rows,
            new_b2b=int(new_b2b),
            new_combo=int(new_combo),
        )
        hits.append(cand)
    return hits


def replay_keys_ok(env, frame, keys, nxt):
    """Second check: env._execute_action on the recovered sequence."""
    b2b, combo = _load_state(env, frame)
    if Keys.HARD_DROP not in [int(k) for k in keys]:
        return False, "no hardDrop"
    _, _, _, _, next_board, _, _, new_hold, new_queue = env._execute_action(
        env._board,
        env._vis_board,
        env._active_piece,
        env._hold_piece,
        list(env._queue),
        np.asarray(keys, dtype=np.int64),
    )
    if env._scorer._b2b != int(nxt["b2b_combo_garbage"][0]):
        return (
            False,
            f"exec b2b {env._scorer._b2b} != {int(nxt['b2b_combo_garbage'][0])}",
        )
    if env._scorer._combo != int(nxt["b2b_combo_garbage"][1]):
        return (
            False,
            f"exec combo {env._scorer._combo} != {int(nxt['b2b_combo_garbage'][1])}",
        )
    ok, _ = _board_match(next_board, nxt["board"])
    if not ok:
        return False, "exec board mismatch"
    expected = (
        new_hold.value,
        [p.value for p in new_queue],
    )
    nxt_hold = int(nxt["pieces"][1])
    nxt_q = [int(p) for p in nxt["pieces"][2:]]
    if expected[0] != nxt_hold:
        return False, f"exec hold {expected[0]} != {nxt_hold}"
    if nxt_q[: len(expected[1])] != expected[1]:
        return False, "exec queue mismatch"
    return True, ""


def _same_round(a, b):
    return (a.get("game"), a.get("round")) == (b.get("game"), b.get("round"))


def recover_replay(frames, env=None, searcher=None):
    if env is None:
        env = _make_env()
    if searcher is None:
        searcher = CB2BSearch()
    finder = env._key_sequence_finder
    rows = []
    for i in range(len(frames) - 1):
        if not _same_round(frames[i], frames[i + 1]):
            rows.append(
                {
                    "i": i,
                    "n_hits": 0,
                    "hits": [],
                    "chosen": None,
                    "exec_ok": None,
                    "exec_err": "",
                    "boundary": True,
                }
            )
            continue
        hits = recover_frame(env, finder, searcher, frames[i], frames[i + 1])
        # Dedup by placement identity.
        uniq = {}
        for h in hits:
            key = (
                h["is_hold"],
                h["rot"],
                h["norm_col"],
                h["landing_row"],
                h["spin"],
            )
            uniq.setdefault(key, h)
        hits = list(uniq.values())
        chosen = hits[0] if len(hits) == 1 else (hits[0] if hits else None)
        exec_ok, exec_err = None, ""
        if chosen is not None:
            exec_ok, exec_err = replay_keys_ok(
                env, frames[i], chosen["keys"], frames[i + 1]
            )
        rows.append(
            {
                "i": i,
                "n_hits": len(hits),
                "hits": hits,
                "chosen": chosen,
                "exec_ok": exec_ok,
                "exec_err": exec_err,
            }
        )
    return rows


def default_out_path(src):
    root, ext = os.path.splitext(src)
    return f"{root}.recovered{ext or '.json'}"


def write_recovered_json(src_frames, rows, dest, mode=None):
    """Copy frames, replace keys on recovered transitions, write a new JSON."""
    out = []
    by_i = {r["i"]: r for r in rows}
    for i, src in enumerate(src_frames):
        frame = dict(src)
        row = by_i.get(i)
        if row is not None and row["chosen"] is not None:
            keys = [int(k) for k in row["chosen"]["keys"]]
            frame["key_sequence"] = keys
            frame["converted_keys"] = _converted(keys)
        out.append(frame)
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    with open(dest, "w") as f:
        json.dump({"mode": mode, "frames": out}, f)
    return dest


def _print_row(row, verbose):
    i = row["i"]
    n = row["n_hits"]
    if row.get("boundary"):
        print(f"  [{i:3d}] ROUND BOUNDARY")
        return
    if n == 0:
        print(f"  [{i:3d}] MISS")
        return
    ch = row["chosen"]
    keys = _converted(ch["keys"])
    flag = "OK" if n == 1 else f"AMBIG x{n}"
    exec_s = (
        " exec=ok"
        if row["exec_ok"]
        else (f" exec=FAIL {row['exec_err']}" if row["exec_ok"] is False else "")
    )
    print(
        f"  [{i:3d}] {flag:10s} hold={int(ch['is_hold'])} "
        f"atk={ch['attack']:.0f} cl={ch['clears']} garb={ch['garb_rows']} "
        f"b2b={ch['new_b2b']} combo={ch['new_combo']} "
        f"keys={keys}{exec_s}"
    )
    if verbose and n > 1:
        for h in row["hits"][1:]:
            print(
                f"         alt hold={int(h['is_hold'])} "
                f"r={h['rot']} c={h['norm_col']} lr={h['landing_row']} "
                f"spin={h['spin']} keys={_converted(h['keys'])}"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_REPLAY,
        help="replay JSON (default: the 2026-07-07 spectate file)",
    )
    ap.add_argument("--limit", type=int, default=None, help="only first N transitions")
    ap.add_argument("--verbose", action="store_true", help="print alternate hits")
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: <input>.recovered.json)",
    )
    args = ap.parse_args(argv)

    with open(args.path) as f:
        data = json.load(f)
    mode, frames = data.get("mode"), data["frames"]
    if args.limit is not None:
        frames = frames[: args.limit + 1]
    print(f"{os.path.basename(args.path)}: {len(frames)} frames")

    rows = recover_replay(frames)
    for row in rows:
        _print_row(row, args.verbose)

    dest = args.out or default_out_path(args.path)
    write_recovered_json(frames, rows, dest, mode)

    bounds = sum(1 for r in rows if r.get("boundary"))
    rows = [r for r in rows if not r.get("boundary")]
    n = len(rows)
    miss = sum(1 for r in rows if r["n_hits"] == 0)
    uniq = sum(1 for r in rows if r["n_hits"] == 1)
    amb = sum(1 for r in rows if r["n_hits"] > 1)
    exec_fail = sum(
        1 for r in rows if r["chosen"] is not None and r["exec_ok"] is False
    )
    exec_ok = sum(1 for r in rows if r["exec_ok"] is True)
    atk = sum(r["chosen"]["attack"] for r in rows if r["chosen"] is not None)
    pieces = sum(1 for r in rows if r["chosen"] is not None)
    print()
    print(
        f"transitions={n} unique={uniq} ambiguous={amb} miss={miss} "
        f"exec_ok={exec_ok} exec_fail={exec_fail} round_boundaries={bounds}"
    )
    if pieces:
        print(
            f"recovered attack={atk:.0f} over {pieces} pieces  APP={atk / pieces:.3f}"
        )
    print(f"wrote {dest}")
    return 0 if miss == 0 and exec_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
