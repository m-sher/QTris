"""Root visit statistics of several search configurations on the same positions.

Usage: uv run python scripts/search_stats.py OUT.json FRAMES.json ARMS.json [eps].
FRAMES is a replay-style frame file (an expert replay, or scripts/onpolicy_frames.py's
output); ARMS maps an arm name to MCTSConfig overrides; every arm sees identical Dirichlet
draws. Records per position the root width, top-1 visit share, visit perplexity, KL of the
visit target from the prior, whether the search moved the prior's argmax, the class of
the chosen move, and agreement with the expert's recorded move where the frame has one."""

import json
import sys

import numpy as np
import tensorflow as tf
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Pieces import PieceType

from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.training._1v1_placement_az import _build_game_pairs, _build_net

out, frames_path, arms_path = sys.argv[1], sys.argv[2], sys.argv[3]
EPS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
ARMS = json.load(open(arms_path))
B = 16
net = _build_net(B, 8, 64, 4, 4, 5)
tf.train.Checkpoint(model=net).restore(
    tf.train.latest_checkpoint("checkpoints/1v1_placement_az")
).expect_partial()
envs = [p[0] for p in _build_game_pairs(B, 5, 50, 15)]
for e in envs:
    e._reset()
searcher = CB2BSearch()
frames = json.load(open(frames_path))["frames"]


def load(env, fr):
    stored = np.asarray(fr["board"], dtype=np.float32)
    b = np.zeros((40, 10), dtype=np.float32)
    rows = min(stored.shape[0], 40)
    b[-rows:] = stored[-rows:]
    env._board = b
    env._vis_board = b.copy()
    pieces = [int(p) for p in fr["pieces"]]
    env._active_piece = env._spawn_piece(PieceType(pieces[0]))
    env._hold_piece = PieceType(pieces[1])
    env._queue = [PieceType(p) for p in pieces[2 : 2 + env._queue_size]]
    env._next_bag = [
        PieceType(p) for p in pieces[2 + env._queue_size : 2 + env._queue_size + 7]
    ]
    b2b, combo, _ = fr["b2b_combo_garbage"]
    env._scorer._b2b = int(b2b)
    env._scorer._combo = int(combo)
    env._garbage_queue = [
        (int(g[0]), 0, env._garbage_push_delay) for g in fr.get("garbage_queue") or []
    ]


def expert_board(env, fr):
    """Post-lock board of the expert's recorded key sequence, or None."""
    if fr.get("used") is False or fr.get("key_sequence") is None or "player" in fr:
        return None
    b2b, combo = env._scorer._b2b, env._scorer._combo
    try:
        res = env._execute_action(
            env._board.copy(),
            env._vis_board.copy(),
            env._active_piece,
            env._hold_piece,
            list(env._queue),
            np.asarray(fr["key_sequence"], dtype=np.int64),
        )
        board = res[4]
    except Exception:
        board = None
    env._scorer._b2b, env._scorer._combo = b2b, combo
    env._active_piece = env._spawn_piece(PieceType(int(fr["pieces"][0])))
    return None if board is None else (np.asarray(board) != 0)


def candidate_boards(env, desc, mask):
    boards = {}
    for slot in np.flatnonzero(mask):
        is_hold, rot, norm_col, landing_row, spin = (int(x) for x in desc[slot])
        if is_hold:
            placed = (
                env._queue[0] if env._hold_piece == PieceType.N else env._hold_piece
            )
        else:
            placed = env._active_piece.piece_type
        nb, clears, attack, new_b2b, _c = searcher.lock_score(
            env._board,
            placed.value,
            rot,
            norm_col,
            landing_row,
            spin,
            env._scorer._b2b,
            env._scorer._combo,
        )
        boards[slot] = (np.asarray(nb) != 0, int(clears), float(attack), int(new_b2b))
    return boards


results = {name: [] for name in ARMS}
for start in range(0, len(frames), B):
    batch = frames[start : start + B]
    for env, fr in zip(envs, batch):
        load(env, fr)
    # candidate boards and the expert's move, once per batch (search-independent)
    from qtris.search.cmcts import CMCTS

    cand_info = []
    engine = CMCTS(len(batch), board_height=40, queue_size=5, max_holes=50, max_len=15)
    try:
        for i in range(len(batch)):
            engine.set_root(i, envs[i])
        nv, req = engine.collect_roots()
        _pi, _c, desc, _d, _rv = engine.result()
        masks = np.array(req[4], dtype=bool)
        tree_ids = list(req[5])
    finally:
        engine.destroy()
    per_tree = {}
    for k in range(nv):
        i = tree_ids[k]
        boards = candidate_boards(envs[i], np.array(desc[i], np.int64), masks[k])
        eb = expert_board(envs[i], batch[i])
        exp_slot = None
        if eb is not None:
            for slot, (nb, *_r) in boards.items():
                if nb.shape == eb.shape and np.array_equal(nb, eb):
                    exp_slot = int(slot)
                    break
        per_tree[i] = dict(boards=boards, exp_slot=exp_slot)
    for name, over in ARMS.items():
        cfg = MCTSConfig(
            num_simulations=256,
            leaves_per_round=8,
            gamma=1.0,
            w_death=1.0,
            dirichlet_eps=EPS,
            **over,
        )
        mcts = PlacementMCTS(net, cfg)
        for env, fr in zip(envs, batch):
            load(env, fr)
        np.random.seed(1000 + start)
        res = mcts.search(envs[: len(batch)], 1.0, np.zeros(len(batch), np.float32))
        # priors from a fresh root eval on the returned observations
        live = [i for i, r in enumerate(res) if not r["dead"]]
        if not live:
            continue
        boards = np.stack([res[i]["board"] for i in live])
        pieces = np.stack([res[i]["pieces"] for i in live])
        bcg = np.stack([res[i]["bcg"] for i in live])
        pls = np.stack([res[i]["cand_placements"] for i in live])
        masks_l = np.stack([res[i]["cand_mask"] for i in live])
        mcts._fullb = B * cfg.leaves_per_round
        logits, _v = mcts._net_eval(boards, pieces, bcg, pls, masks_l)
        for j, i in enumerate(live):
            r = res[i]
            m = masks_l[j]
            lg = np.where(m, logits[j], -1e9)
            pr = np.exp(lg - lg.max())
            pr = pr / pr.sum()
            pi = np.asarray(r["pi"], np.float64)
            legal = np.flatnonzero(m)
            pil = pi[legal]
            prl = pr[legal]
            pil = pil / pil.sum() if pil.sum() > 0 else pil
            info = per_tree.get(i, {})
            boards_i = info.get("boards", {})
            fr = batch[i]
            top = int(legal[np.argmax(pil)])
            ent = float(-(pil[pil > 0] * np.log(pil[pil > 0])).sum())
            kl = float(
                (
                    pil[pil > 0]
                    * np.log(pil[pil > 0] / np.maximum(prl[pil > 0], 1e-12))
                ).sum()
            )
            b2b = int(fr["b2b_combo_garbage"][0])
            results[name].append(
                dict(
                    idx=start + i,
                    n_legal=int(m.sum()),
                    visited=int((np.asarray(r["counts"])[legal] > 0).sum()),
                    top1=float(pil.max()),
                    perplexity=float(np.exp(ent)),
                    prior_entropy=float(-(prl[prl > 0] * np.log(prl[prl > 0])).sum()),
                    kl=kl,
                    argmax_moves=bool(top != int(legal[np.argmax(prl)])),
                    low_prior_mass=float(pil[prl < 0.015].sum()),
                    v_root=float(r["value"]),
                    v_search=float(r["v_search"]),
                    pre_b2b=b2b,
                    pending=int(fr["b2b_combo_garbage"][2]),
                    h=int(np.count_nonzero(np.asarray(fr["board"]).any(1))),
                    exp_slot=info.get("exp_slot"),
                    exp_match=(
                        info.get("exp_slot") is not None and top == info.get("exp_slot")
                    ),
                    exp_prior=(
                        float(pr[info["exp_slot"]])
                        if info.get("exp_slot") is not None
                        else None
                    ),
                    exp_pi=(
                        float(pi[info["exp_slot"]])
                        if info.get("exp_slot") is not None
                        else None
                    ),
                    top_clears=int(boards_i[top][1]) if top in boards_i else None,
                    top_difficult=bool(boards_i[top][3] == b2b + 1)
                    if top in boards_i
                    else None,
                    top_break=bool(boards_i[top][1] > 0 and boards_i[top][3] == -1)
                    if top in boards_i
                    else None,
                )
            )
    print("batch", start, flush=True)
json.dump(results, open(out, "w"))
for name, rows in results.items():
    print(name, "n", len(rows))
