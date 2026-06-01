"""Neural-guided placement search (fusion-style): the policy gates/priors candidate
placements and the state-only value evaluates the resulting boards via lookahead.

Purely inference over a trained `PlacementPolicyValueNet` - no new training. Each
node clones the env (sharing its stateless C handles) and calls the real `_step` to
simulate a placement, so child states match the training distribution exactly.
"""

import copy
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from TetrisEnv.Moves import Keys
from qtris.data.placement_features import build_placement_inference

ROW_NORM = 23  # board height - 1 (24-row board)
_FORCED_DROP = np.array([Keys.START, Keys.HARD_DROP] + [Keys.PAD] * 13, dtype=np.int64)


@dataclass
class SearchConfig:
    depth: int = 2
    beam_width: int = 8
    gate_k: int = 8
    policy_bonus_weight: float = 0.1
    enum_depth: int = 2
    enum_beam: int = 512
    max_len: int = 15


def clone_sim_env(env):
    """Deep-copy the env for simulation, sharing the two stateless ctypes C handles
    (which `deepcopy` can't pickle). The clone runs the real `_step` with pathfinding
    off (skip the costly obs pathfinder) and garbage off (no speculative garbage;
    pending garbage carried by the copy still resolves deterministically)."""
    ksf, hf = env._key_sequence_finder, env._hole_finder
    env._key_sequence_finder = None
    env._hole_finder = None
    try:
        clone = copy.deepcopy(env)
    finally:
        env._key_sequence_finder = ksf
        env._hole_finder = hf
    clone._key_sequence_finder = ksf
    clone._hole_finder = hf
    clone._pathfinding = False
    clone._garbage_chance = 0.0
    return clone


def net_input_from_env(env):
    """Build the (board, pieces, bcg) net input from env attributes, batched to (1, ...).
    Mirrors `_create_observation` exactly."""
    pieces = np.array(
        [env._active_piece.piece_type.value, env._hold_piece.value]
        + [p.value for p in env._queue],
        dtype=np.int64,
    )
    board = env._board[None, ..., None].astype(np.float32)  # (1,24,10,1)
    bcg = np.array(
        [env._scorer._b2b, env._scorer._combo, env._get_total_garbage()],
        dtype=np.float32,
    )[None]  # (1,3)
    return board, pieces[None], bcg


def enumerate_node(env, searcher, cfg):
    """Enumerate candidate placements for the node via the C search; return
    (placements[128,18], mask[128], sequences[128,max_len]) or None if dead."""
    queue = np.array([p.value for p in env._queue], dtype=np.int32)
    best_action, _best_seq, ca, cs, cseq, cr = searcher.search_with_scores(
        board=env._board,
        active_piece=env._active_piece.piece_type.value,
        hold_piece=env._hold_piece.value,
        queue=queue,
        b2b=int(env._scorer._b2b),
        combo=int(env._scorer._combo),
        total_garbage=int(env._get_total_garbage()),
        garbage_push_delay=env._garbage_push_delay,
        search_depth=cfg.enum_depth,
        beam_width=cfg.enum_beam,
        max_len=cfg.max_len,
    )
    if best_action < 0 or len(ca) == 0:
        return None
    return build_placement_inference(
        ca,
        cs,
        cr,
        cseq,
        active_piece=env._active_piece.piece_type.value,
        hold_piece=env._hold_piece.value,
        queue0=int(queue[0]) if len(queue) else 0,
        row_norm=ROW_NORM,
        max_len=cfg.max_len,
    )


def _policy_value_batch(net, boards, pieces, bcgs, placements, masks):
    logits, value = net(
        (
            tf.constant(np.concatenate(boards), tf.float32),
            tf.constant(np.concatenate(pieces), tf.int64),
            tf.constant(np.concatenate(bcgs), tf.float32),
            tf.constant(np.stack(placements), tf.float32),
            tf.constant(np.stack(masks), tf.bool),
        ),
        training=False,
    )
    return logits.numpy(), value.numpy()[:, 0]


def _value_batch(net, boards, pieces, bcgs):
    piece_dec, _ = net.process_obs(
        (
            tf.constant(np.concatenate(boards), tf.float32),
            tf.constant(np.concatenate(pieces), tf.int64),
            tf.constant(np.concatenate(bcgs), tf.float32),
        ),
        training=False,
    )
    return net.score_value(piece_dec, training=False).numpy()[:, 0]


def _gate(logits, mask, gate_k):
    """Indices of the top-gate_k legal candidates by policy logit."""
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return legal
    return legal[np.argsort(logits[legal])[::-1]][:gate_k]


def search_best_move(env, net, searcher, cfg):
    """Return the key sequence (np.int64[max_len]) of the root placement whose
    best lookahead board (value + policy prior) is highest."""
    root = clone_sim_env(env)
    enum = enumerate_node(root, searcher, cfg)
    if enum is None:
        return _FORCED_DROP.copy()
    placements, mask, seqs = enum
    board, pieces, bcg = net_input_from_env(root)
    logits, _ = _policy_value_batch(net, [board], [pieces], [bcg], [placements], [mask])
    frontier = [
        {
            "env": root,
            "logits": logits[0],
            "seqs": seqs,
            "mask": mask,
            "root_move": None,
        }
    ]
    best_by_root = {}

    for ply in range(cfg.depth):
        is_leaf = ply == cfg.depth - 1
        kids = []  # (env, root_move, parent_logit, terminal)
        for node in frontier:
            for slot in _gate(node["logits"], node["mask"], cfg.gate_k):
                seq = node["seqs"][slot]
                if not np.any(seq == Keys.HARD_DROP):
                    continue
                child = clone_sim_env(node["env"])
                ts = child._step(seq.astype(np.int64))
                rm = node["root_move"] if node["root_move"] is not None else seq.copy()
                kids.append(
                    (child, rm, float(node["logits"][slot]), bool(ts.is_last()))
                )
        if not kids:
            break

        boards = [net_input_from_env(c[0])[0] for c in kids]
        pieces_l = [net_input_from_env(c[0])[1] for c in kids]
        bcgs = [net_input_from_env(c[0])[2] for c in kids]

        if is_leaf:
            values = _value_batch(net, boards, pieces_l, bcgs)
            for (_, rm, plogit, terminal), v in zip(kids, values):
                score = (
                    -1e30 if terminal else float(v) + cfg.policy_bonus_weight * plogit
                )
                k = rm.tobytes()
                best_by_root[k] = (
                    (max(best_by_root[k][0], score), rm)
                    if k in best_by_root
                    else (score, rm)
                )
        else:
            enums = [enumerate_node(c[0], searcher, cfg) for c in kids]
            keep = [i for i, e in enumerate(enums) if e is not None and not kids[i][3]]
            new_frontier = []
            if keep:
                pls = [enums[i][0] for i in keep]
                msks = [enums[i][1] for i in keep]
                logits_b, values_b = _policy_value_batch(
                    net,
                    [boards[i] for i in keep],
                    [pieces_l[i] for i in keep],
                    [bcgs[i] for i in keep],
                    pls,
                    msks,
                )
                for j, i in enumerate(keep):
                    child, rm, plogit, _ = kids[i]
                    score = float(values_b[j]) + cfg.policy_bonus_weight * plogit
                    k = rm.tobytes()
                    best_by_root[k] = (
                        (max(best_by_root[k][0], score), rm)
                        if k in best_by_root
                        else (score, rm)
                    )
                    new_frontier.append(
                        {
                            "env": child,
                            "logits": logits_b[j],
                            "seqs": enums[i][2],
                            "mask": msks[j],
                            "root_move": rm,
                            "_score": score,
                        }
                    )
            # terminal / dead children still score their root move (very low)
            for i, (child, rm, plogit, terminal) in enumerate(kids):
                if enums[i] is None or terminal:
                    k = rm.tobytes()
                    if k not in best_by_root:
                        best_by_root[k] = (-1e30, rm)
            frontier = sorted(new_frontier, key=lambda n: n["_score"], reverse=True)[
                : cfg.beam_width
            ]
            if not frontier:
                break

    if not best_by_root:
        return _FORCED_DROP.copy()
    _, best_rm = max(best_by_root.values(), key=lambda sv: sv[0])
    return best_rm.astype(np.int64)
