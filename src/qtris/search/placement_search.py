"""Env helpers shared by the AlphaZero trainer, the eval scripts and the demos."""

import copy

import numpy as np

from TetrisEnv.Moves import Keys
from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import apply_four_wide_walls


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


def descriptor_key_sequence(env, desc, max_len=None):
    """Key sequence for a placement descriptor `(is_hold, rot, norm_col, landing_row,
    spin)`, matched against the multi-landing enum on all four placement fields.
    Returns a START+HARD_DROP sequence when the pathfinder yields no match.
    """
    is_hold, rot, norm_col, landing_row, spin = (int(x) for x in desc)
    ml = int(max_len if max_len is not None else env._max_len)
    if is_hold:
        piece = (
            env._spawn_piece(env._hold_piece)
            if env._hold_piece != PieceType.N
            else env._spawn_piece(env._queue[0])
        )
    else:
        piece = env._active_piece
    finder = env._key_sequence_finder
    if not hasattr(finder, "find_unique_placements"):
        # Finder without find_unique_placements: dense lookup keyed by
        # (rot, norm_col, spin); landing_row is not matched.
        seqs, lrs = finder.find_placement_candidates(
            board=env._board, piece=piece, max_len=ml, is_hold=bool(is_hold)
        )
        idx = rot * 40 + norm_col * 4 + spin
        if 0 <= idx < 160 and lrs[idx] >= 0:
            return np.asarray(seqs[idx], dtype=np.int64)
    else:
        rots, ncols, lrs, spins, seqs = finder.find_unique_placements(
            board=env._board,
            piece=piece,
            max_len=ml,
            is_hold=bool(is_hold),
            max_out=512,
            with_sequences=True,
        )
        for i in range(len(rots)):
            if (
                int(rots[i]) == rot
                and int(ncols[i]) == norm_col
                and int(lrs[i]) == landing_row
                and int(spins[i]) == spin
            ):
                return np.asarray(seqs[i], dtype=np.int64)
    forced = np.full(ml, Keys.PAD, dtype=np.int64)
    forced[0], forced[1] = Keys.START, Keys.HARD_DROP
    return forced


def placement_step(env, searcher, desc):
    """Step `env` by a placement descriptor `(is_hold, rot, norm_col, landing_row, spin)`,
    locking + scoring via the C core (`searcher.lock_score`) and reusing the env's own
    garbage / stats / shaping-reward path exactly as `_step`. Mutates `env` in place and
    returns `(total_reward, attack, clears, died)`. Equivalent to `_step(key_sequence)`."""
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
    if env._four_wide:
        apply_four_wide_walls(board)

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
