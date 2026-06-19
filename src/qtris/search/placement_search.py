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
from TetrisEnv.Pieces import PieceType
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
    """Lightweight clone for simulation: shallow-copy the env (sharing the stateless C
    handles, rotation tables, and specs) and copy only the state a sim `_step` mutates
    in place - scorer, garbage queue, bag, and the two RNGs. The board / active piece /
    queue refs are safe to share because `_step` rebuilds them (it deep-copies internally
    before writing) rather than mutating in place. Pathfinding (costly obs pathfinder) and
    speculative garbage are disabled; pending garbage still resolves deterministically."""
    clone = copy.copy(env)
    clone._scorer = copy.copy(env._scorer)
    clone._garbage_queue = list(env._garbage_queue)
    clone._next_bag = list(env._next_bag)
    clone._tetrio_rng = copy.deepcopy(env._tetrio_rng)
    clone._random = copy.deepcopy(env._random)
    clone._pathfinding = False
    clone._garbage_chance = 0.0
    return clone


def placement_step(env, searcher, desc):
    """Step `env` by a placement descriptor `(is_hold, rot, norm_col, landing_row, spin)`,
    locking + scoring via the C core (`searcher.lock_score`) and reusing the env's own
    garbage / stats / shaping-reward path exactly as `_step`. Mutates `env` in place and
    returns `(total_reward, attack, clears, died)`. Verified equivalent to `_step(key_sequence)`
    by the placement parity gate, so committing real moves by descriptor matches `_step`."""
    is_hold, rot, norm_col, landing_row, spin = (int(x) for x in desc)
    env._step_num += 1
    pre_b2b = env._scorer._b2b

    queue = list(env._queue)
    active = env._active_piece
    hold = env._hold_piece
    if is_hold:
        placed_type = queue.pop(0) if hold == PieceType.N else hold
        new_hold = active.piece_type
    else:
        placed_type = active.piece_type
        new_hold = hold

    board, clears, attack, new_b2b, new_combo = searcher.lock_score(
        env._board,
        placed_type.value,
        rot,
        norm_col,
        landing_row,
        spin,
        env._scorer._b2b,
        env._scorer._combo,
    )
    env._scorer._b2b = new_b2b
    env._scorer._combo = new_combo
    next_active = env._spawn_piece(queue.pop(0))
    top_out = env._is_top_out(board)

    vis = env._vis_board
    if attack > 0:
        env._remove_attack_from_garbage_queue(attack)
    if env._auto_push_garbage and clears == 0:
        env._tick_garbage_timers()
        board, vis, _ = env._push_garbage_to_board(board, vis)
    env._add_to_garbage_queue()
    if env._auto_push_garbage and env._garbage_push_delay == 0:
        while env._garbage_queue:
            board, vis, pushed = env._push_garbage_to_board(board, vis)
            if not pushed:
                break

    height_val, holes_val, skyline_val, bumpy_val = env._board_stats(board)
    attack_reward = env._attack_reward * attack
    if env._use_shaping:
        current_phi = env._calculate_potential(
            env._scorer._b2b,
            env._scorer._combo,
            height_val,
            holes_val,
            skyline_val,
            bumpy_val,
        )
        shaping_reward = env._gamma * current_phi - env._last_phi
    else:
        current_phi = 0.0
        shaping_reward = 0.0
    extension_bonus = (
        env._b2b_extend_flat + env._b2b_extend_scale * max(0, env._scorer._b2b)
        if env._scorer._b2b > pre_b2b
        else 0.0
    )
    exceeded_holes = holes_val > env._max_holes if env._max_holes is not None else False
    garbage_top_out = env._is_top_out(board)
    died = top_out or exceeded_holes or garbage_top_out
    total_reward = (
        attack_reward
        + shaping_reward
        + extension_bonus
        + (env._death_penalty if died else 0.0)
    )

    if env._auto_fill_queue:
        queue = env._fill_queue(queue)
    env._board = board
    env._vis_board = vis
    env._active_piece = next_active
    env._hold_piece = new_hold
    env._queue = queue
    env._last_phi = current_phi
    env._episode_ended = died or (
        env._step_num >= env._max_steps if env._max_steps else False
    )
    return float(total_reward), float(attack), int(clears), died


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
    logits, value = net.policy_value(
        (
            tf.constant(np.concatenate(boards), tf.float32),
            tf.constant(np.concatenate(pieces), tf.int64),
            tf.constant(np.concatenate(bcgs), tf.float32),
            tf.constant(np.stack(placements), tf.float32),
            tf.constant(np.stack(masks), tf.bool),
        )
    )
    return logits.numpy(), value.numpy()[:, 0]


def _value_batch(net, boards, pieces, bcgs):
    return net.state_value(
        tf.constant(np.concatenate(boards), tf.float32),
        tf.constant(np.concatenate(pieces), tf.int64),
        tf.constant(np.concatenate(bcgs), tf.float32),
    ).numpy()[:, 0]


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
