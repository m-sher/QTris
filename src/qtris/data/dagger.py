"""DAgger-style data collection for the Tetris policy.

The loop rolls a trained policy forward in env and labels each visited state
with the beam search's dense action-indexed target (the same schema gen_ar
stores):

  * Roll the trained policy forward in env (greedy decode under valid-sequence
    masking).
  * At each visited state, query the beam search for the per-action scores +
    key-sequences and scatter them into a dense action-indexed target.
  * Step the env with the POLICY's choice (the DAgger invariant) - this shifts
    the visited-state distribution toward what the policy sees in deployment,
    the canonical fix for compounding error in BC on long-horizon games
    (Ross & Bagnell, 2010).
  * Record (state, dense search target).

``family`` selects which policy checkpoint drives the rollout and which target
schema is stored: ar/flat use gen_ar's dense 320-action schema; placement uses
gen_placement's 128-slot placement schema. Within a family the DAgger output
matches the pretrain dataset, so transitions accumulate across BC + DAgger rounds.
"""

import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from qtris.models.ar.model import PolicyModel
from qtris.models.flat.model import FlatPolicyModel
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.data.gen_ar import NUM_ACTIONS, dense_target
from qtris.data.placement_features import (
    CANDIDATE_CAPACITY,
    PLACEMENT_FEATURE_DIM,
    build_placement_inference,
    build_placement_target,
)

# Fields of a placement state record (net inputs + oracle search inputs), in save order.
_STATE_FIELDS = {
    "board": np.float32,
    "pieces": np.int64,
    "bcg": np.float32,
    "board_occ": np.float32,
    "active": np.int32,
    "hold": np.int32,
    "queue": np.int32,
    "b2b": np.int32,
    "combo": np.int32,
    "total_garbage": np.int32,
    "push_delay": np.int32,
}


def _state_record(env):
    """Placement state dict from a live env: the net inputs (board/pieces/bcg, identical to
    `_create_observation`) plus the oracle search inputs (board_occ/active/hold/queue/b2b/combo/
    total_garbage/push_delay). label_placement_states consumes exactly these keys, so a record
    from a DAgger rollout and one dumped during AZ self-play are interchangeable."""
    pieces = np.array(
        [env._active_piece.piece_type.value, env._hold_piece.value]
        + [p.value for p in env._queue],
        dtype=np.int64,
    )
    return {
        # net input: the model-visible slice (bottom 24); board_occ keeps the full board for the oracle
        "board": env._board[-24:][..., None].astype(np.float32),
        "pieces": pieces,
        "bcg": np.array(
            [env._scorer._b2b, env._scorer._combo, env._get_total_garbage()],
            dtype=np.float32,
        ),
        "board_occ": env._board.copy().astype(np.float32),
        "active": int(env._active_piece.piece_type.value),
        "hold": int(env._hold_piece.value),
        "queue": np.array([p.value for p in env._queue], dtype=np.int32),
        "b2b": int(env._scorer._b2b),
        "combo": int(env._scorer._combo),
        "total_garbage": int(env._get_total_garbage()),
        "push_delay": int(env._garbage_push_delay),
    }


def save_states(states, path):
    """Save a list of state records as a tf.data dataset shard (the codebase's dataset format).
    Phase 2 (label_placement_states) reads these back to label with the oracle."""
    cols = {}
    for k, dt in _STATE_FIELDS.items():
        vals = [s[k] for s in states]
        cols[k] = (
            np.stack(vals).astype(dt)
            if np.ndim(vals[0]) > 0
            else np.array(vals, dtype=dt)
        )
    tf.data.Dataset.from_tensor_slices(cols).save(str(path))


def _shard_paths(states_dir):
    """Sorted `shard_*` tf.data dirs under states_dir."""
    states_dir = str(states_dir)
    return sorted(
        os.path.join(states_dir, d)
        for d in os.listdir(states_dir)
        if d.startswith("shard_")
    )


def _load_shard(shard_dir):
    """Load one tf.data state shard into a list of state dicts."""
    ds = tf.data.Dataset.load(shard_dir)
    cols = {k: v.numpy() for k, v in next(iter(ds.batch(10_000_000))).items()}
    n = len(cols["active"])
    return [{k: cols[k][i] for k in cols} for i in range(n)]


def collect_dagger(
    p_model,
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    headless=False,
    log_every=1000,
):
    """Roll the policy forward; label each visited state with the search target.

    The env is stepped with the policy's greedy choice (under valid-sequence
    masking) - the DAgger invariant that shifts the visited-state distribution
    toward deployment. The beam search labels each state with the same dense
    action-indexed target as gen_ar; the search's own best move does not drive
    transitions.
    """
    env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_steps=max_steps_env,
        max_len=max_len,
        pathfinding=True,
        seed=seed,
        idx=0,
        garbage_chance=garbage_chance,
        garbage_min=garbage_min,
        garbage_max=garbage_max,
        garbage_push_delay=garbage_push_delay,
        auto_push_garbage=True,
        auto_fill_queue=True,
        num_row_tiers=num_row_tiers,
    )

    obs, _ = env.reset()
    searcher = CB2BSearch()

    transitions = []
    beam_dead = 0
    deaths = 0
    max_b2b = 0
    policy_disagrees = 0
    total_attack = 0.0
    pieces_placed = 0

    pbar = tqdm(range(num_steps), disable=headless, desc="dagger", unit="step")
    for step in pbar:
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)
        valid_sequences = obs["sequences"].astype(np.int64)

        # Policy's greedy choice under valid-sequence masking (drives the env).
        # PolicyModel and FlatPolicyModel share predict(): both return the
        # selected key-sequence as the first tuple element.
        b_in = tf.constant(board[None, ...], dtype=tf.float32)
        p_in = tf.constant(pieces[None, ...], dtype=tf.int64)
        g_in = tf.constant(bcg[None, ...], dtype=tf.float32)
        vs_in = tf.constant(valid_sequences[None, ...], dtype=tf.int64)
        policy_seq, _, _, _ = p_model.predict(
            (b_in, p_in, g_in),
            greedy=True,
            valid_sequences=vs_in,
            temperature=1.0,
        )
        policy_seq = policy_seq.numpy()[0].astype(np.int64)

        # Beam's dense action-indexed target for this state (the label).
        best_action, best_seq, cand_actions, cand_scores, cand_seqs, _cand_rows = (
            searcher.search_with_scores(
                board=env._board,
                active_piece=env._active_piece.piece_type.value,
                hold_piece=env._hold_piece.value,
                queue=np.array([p.value for p in env._queue], dtype=np.int32),
                b2b=int(env._scorer._b2b),
                combo=int(env._scorer._combo),
                total_garbage=int(env._get_total_garbage()),
                garbage_push_delay=env._garbage_push_delay,
                search_depth=search_depth,
                beam_width=beam_width,
                max_len=max_len,
            )
        )

        if best_action < 0 or len(cand_scores) == 0:
            # No labelable placement - treat as terminal and reset.
            beam_dead += 1
            deaths += 1
            obs, _ = env.reset()
            continue

        seqs, scores = dense_target(cand_actions, cand_scores, cand_seqs, max_len)
        transitions.append((board, pieces, bcg, seqs, scores))

        # Step env with the POLICY's choice (DAgger invariant).
        obs, _reward, terminated, truncated, info = env.step(policy_seq)
        total_attack += float(info["attack"])
        pieces_placed += 1
        if not np.array_equal(policy_seq, best_seq):
            policy_disagrees += 1
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if terminated or truncated:
            deaths += 1
            obs, _ = env.reset()

        if (step + 1) % log_every == 0:
            disagree_rate = 100.0 * policy_disagrees / (step + 1)
            app = total_attack / max(pieces_placed, 1)
            stats = (
                f"transitions={len(transitions)} beam_dead={beam_dead} "
                f"deaths={deaths} max_b2b={max_b2b} app={app:.3f} "
                f"policy≠beam={policy_disagrees} ({disagree_rate:.1f}%)"
            )
            if headless:
                print(f"Step {step + 1}/{num_steps} | {stats}", flush=True)
            else:
                pbar.set_postfix_str(stats)

    app = total_attack / max(pieces_placed, 1)
    return transitions, beam_dead, deaths, max_b2b, policy_disagrees, app


def rollout_placement_states(
    p_model,
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    searcher,
    headless=False,
    log_every=1000,
):
    """Phase 1: roll the placement policy forward (the DAgger invariant) and record each
    visited state. A state dict holds the net inputs (board/pieces/bcg) and the oracle search
    inputs (board_occ/active/hold/queue/b2b/combo/total_garbage/push_delay), so phase 2 can be
    labeled later from any source. The oracle is run here only to supply the candidate set the
    policy ranks to drive the env; its output is cached on the state so phase 2 need not
    re-search these states. Returns (states, stats)."""
    env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_steps=max_steps_env,
        max_len=max_len,
        pathfinding=True,
        seed=seed,
        idx=0,
        garbage_chance=garbage_chance,
        garbage_min=garbage_min,
        garbage_max=garbage_max,
        garbage_push_delay=garbage_push_delay,
        auto_push_garbage=True,
        auto_fill_queue=True,
        num_row_tiers=num_row_tiers,
    )

    obs, _ = env.reset()

    states = []
    beam_dead = 0
    deaths = 0
    max_b2b = 0
    policy_disagrees = 0
    total_attack = 0.0
    pieces_placed = 0

    pbar = tqdm(range(num_steps), disable=headless, desc="dagger rollout", unit="step")
    for step in pbar:
        rec = _state_record(env)

        best_action, best_seq, cand_actions, cand_scores, cand_seqs, cand_rows = (
            searcher.search_with_scores(
                board=rec["board_occ"],
                active_piece=rec["active"],
                hold_piece=rec["hold"],
                queue=rec["queue"],
                b2b=rec["b2b"],
                combo=rec["combo"],
                total_garbage=rec["total_garbage"],
                garbage_push_delay=rec["push_delay"],
                search_depth=search_depth,
                beam_width=beam_width,
                max_len=max_len,
            )
        )

        if best_action < 0 or len(cand_scores) == 0:
            beam_dead += 1
            deaths += 1
            obs, _ = env.reset()
            continue

        states.append(
            {
                **rec,
                "cand_actions": cand_actions,
                "cand_scores": cand_scores,
                "cand_rows": cand_rows,
            }
        )

        # Policy ranks the same candidates; its choice drives the env (DAgger invariant).
        row_norm = rec["board_occ"].shape[0] - 1
        infer_pl, infer_mask, infer_seqs = build_placement_inference(
            cand_actions,
            cand_scores,
            cand_rows,
            cand_seqs,
            rec["active"],
            rec["hold"],
            int(rec["queue"][0]),
            row_norm,
            max_len,
        )
        policy_seq, _, _, _ = p_model.predict(
            (
                tf.constant(rec["board"][None], dtype=tf.float32),
                tf.constant(rec["pieces"][None], dtype=tf.int64),
                tf.constant(rec["bcg"][None], dtype=tf.float32),
                tf.constant(infer_pl[None], dtype=tf.float32),
                tf.constant(infer_mask[None], dtype=tf.bool),
            ),
            greedy=True,
            cand_sequences=tf.constant(infer_seqs[None], dtype=tf.int64),
            temperature=1.0,
        )
        policy_seq = policy_seq.numpy()[0].astype(np.int64)

        obs, _reward, terminated, truncated, info = env.step(policy_seq)
        total_attack += float(info["attack"])
        pieces_placed += 1
        if not np.array_equal(policy_seq, best_seq):
            policy_disagrees += 1
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if terminated or truncated:
            deaths += 1
            obs, _ = env.reset()

        if (step + 1) % log_every == 0:
            disagree_rate = 100.0 * policy_disagrees / (step + 1)
            app = total_attack / max(pieces_placed, 1)
            stats = (
                f"states={len(states)} beam_dead={beam_dead} "
                f"deaths={deaths} max_b2b={max_b2b} app={app:.3f} "
                f"policy≠beam={policy_disagrees} ({disagree_rate:.1f}%)"
            )
            if headless:
                print(f"Step {step + 1}/{num_steps} | {stats}", flush=True)
            else:
                pbar.set_postfix_str(stats)

    app = total_attack / max(pieces_placed, 1)
    return states, {
        "beam_dead": beam_dead,
        "deaths": deaths,
        "max_b2b": max_b2b,
        "policy_disagrees": policy_disagrees,
        "app": app,
    }


def label_placement_states(
    searcher, states, search_depth, beam_width, max_len, progress=False
):
    """Phase 2: run the oracle on collected placement states and build the 128-slot target.

    Source-agnostic - `states` may come from rollout_placement_states OR from AZ self-play
    (saved with the same keys). Each state is a dict of the oracle search inputs (board_occ,
    active, hold, queue, b2b, combo, total_garbage, push_delay) plus the net inputs (board,
    pieces, bcg). A state that already carries the oracle output (cand_actions/cand_scores/
    cand_rows, e.g. from a rollout) reuses it; otherwise the oracle is run here. Returns
    (transitions, beam_dead); each transition is (board, pieces, bcg, placements_t, scores_t)."""
    transitions = []
    beam_dead = 0
    for s in tqdm(states, desc="oracle label", unit="state", disable=not progress):
        cand_scores = s.get("cand_scores")
        if cand_scores is not None:
            cand_actions, cand_rows = s["cand_actions"], s["cand_rows"]
        else:
            best_action, _bseq, cand_actions, cand_scores, _cseq, cand_rows = (
                searcher.search_with_scores(
                    board=s["board_occ"],
                    active_piece=s["active"],
                    hold_piece=s["hold"],
                    queue=s["queue"],
                    b2b=s["b2b"],
                    combo=s["combo"],
                    total_garbage=s["total_garbage"],
                    garbage_push_delay=s["push_delay"],
                    search_depth=search_depth,
                    beam_width=beam_width,
                    max_len=max_len,
                )
            )
            if best_action < 0:
                beam_dead += 1
                continue
        if len(cand_scores) == 0:
            beam_dead += 1
            continue
        row_norm = s["board_occ"].shape[0] - 1
        placements_t, scores_t = build_placement_target(
            cand_actions,
            cand_scores,
            cand_rows,
            s["active"],
            s["hold"],
            int(s["queue"][0]),
            row_norm,
        )
        transitions.append((s["board"], s["pieces"], s["bcg"], placements_t, scores_t))
    return transitions, beam_dead


def collect_dagger_placement(
    p_model,
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    headless=False,
    log_every=1000,
):
    """Placement DAgger as the two phases composed: roll the policy to collect states, then
    label them with the oracle. The same searcher is shared; rollout states carry their cached
    oracle output so labeling does not re-search them."""
    searcher = CB2BSearch()
    states, stats = rollout_placement_states(
        p_model,
        seed,
        num_steps,
        search_depth,
        beam_width,
        queue_size,
        max_len,
        max_holes,
        max_steps_env,
        garbage_chance,
        garbage_min,
        garbage_max,
        garbage_push_delay,
        num_row_tiers,
        searcher,
        headless=headless,
        log_every=log_every,
    )
    transitions, label_beam_dead = label_placement_states(
        searcher, states, search_depth, beam_width, max_len
    )
    return (
        transitions,
        stats["beam_dead"] + label_beam_dead,
        stats["deaths"],
        stats["max_b2b"],
        stats["policy_disagrees"],
        stats["app"],
    )


def _build_ar_model(args):
    p_model = PolicyModel(
        batch_size=1,
        piece_dim=args.piece_dim,
        key_dim=args.key_dim,
        depth=args.depth,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        output_dim=args.key_dim,
    )
    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(args.queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(args.max_len,), dtype=tf.int64),
        )
    )
    return p_model


def _build_flat_model(args):
    num_sequences = 160 * args.num_row_tiers
    p_model = FlatPolicyModel(
        batch_size=1,
        piece_dim=args.piece_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_sequences=num_sequences,
    )
    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(args.queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    return p_model


def _build_placement_model(args):
    p_model = PlacementPolicyValueNet(
        batch_size=1,
        piece_dim=args.piece_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
    )
    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(args.queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(None, PLACEMENT_FEATURE_DIM), dtype=tf.float32),
            keras.Input(shape=(None,), dtype=tf.bool),
        )
    )
    return p_model


def _load_existing(dataset_path, is_placement):
    """Load an existing dataset for append. Returns (existing_dict | None, count); None when the
    path is absent or its schema doesn't match this family."""
    if not os.path.exists(dataset_path):
        return None, 0
    try:
        existing_ds = tf.data.Dataset.load(dataset_path)
        existing = {
            k: v.numpy() for k, v in next(iter(existing_ds.batch(10_000_000))).items()
        }
    except Exception:
        print("Existing dataset load failed, starting fresh", flush=True)
        return None, 0
    count = len(existing.get("cand_scores", []))
    if is_placement:
        cp = existing.get("cand_placements")
        schema_ok = cp is not None and cp.shape[1:] == (
            CANDIDATE_CAPACITY,
            PLACEMENT_FEATURE_DIM,
        )
    else:
        schema_ok = (
            "cand_scores" in existing
            and existing["cand_scores"].shape[1] == NUM_ACTIONS
        )
    if not schema_ok:
        print(
            "Existing dataset is an older/incompatible schema for this family - "
            "starting fresh.",
            flush=True,
        )
        return None, 0
    print(f"Found existing dataset with {count} transitions", flush=True)
    return existing, count


def _merge_and_save(new_transitions, existing, dataset_path, is_placement):
    """Stack new transitions, append to existing, and save the dataset."""
    boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
    pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
    bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
    cand_scores = np.stack([t[4] for t in new_transitions]).astype(np.float32)
    if is_placement:
        label_key = "cand_placements"
        label = np.stack([t[3] for t in new_transitions]).astype(np.float32)
    else:
        label_key = "cand_sequences"
        label = np.stack([t[3] for t in new_transitions]).astype(np.int8)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        label = np.concatenate([existing[label_key], label])
        cand_scores = np.concatenate([existing["cand_scores"], cand_scores])
        print(
            f"Combined: {len(existing['cand_scores'])} existing + "
            f"{len(new_transitions)} new = {len(cand_scores)} total",
            flush=True,
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "boards": boards,
            "pieces": pieces,
            "b2b_combo_garbage": bcg,
            label_key: label,
            "cand_scores": cand_scores,
        }
    )
    # Write to a temp dir and swap, so an interrupted rewrite can't destroy the prior dataset
    # (matters when the relabel deletes input shards after each merge).
    tmp = str(dataset_path) + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    dataset.save(tmp)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.rename(tmp, dataset_path)
    print(f"Saved {len(cand_scores)} transitions to {dataset_path}", flush=True)


def main_label(cli_args):
    """Oracle-predict mode: load placement states saved during AZ self-play and label them with
    the oracle, writing the same dataset the fused DAgger path does. No rollout."""
    from qtris.config import ModelConfig

    m = ModelConfig()
    states_dir = cli_args.label_states
    dataset_path = (
        str(cli_args.output) if cli_args.output else "datasets/tetris_oracle_placement"
    )
    # Recover an orphaned temp dataset from a rewrite interrupted mid-swap (see _merge_and_save).
    tmp = dataset_path + ".tmp"
    if not os.path.exists(dataset_path) and os.path.exists(tmp):
        os.rename(tmp, dataset_path)

    # Label shard-by-shard; each labeled shard's transitions are merged into the dataset, then the
    # shard is deleted - so an interrupted run resumes from the remaining shards (no re-labeling,
    # no duplicates). --save-every is how many shards to label per dataset write.
    shards_per_flush = max(1, getattr(cli_args, "save_every", 1))
    progress = not getattr(cli_args, "headless", False)
    shard_dirs = _shard_paths(states_dir)
    if not shard_dirs:
        print(f"No shards in {states_dir}; dataset unchanged.", flush=True)
        return 0
    print(f"Labeling {len(shard_dirs)} shard(s) from {states_dir}", flush=True)

    searcher = CB2BSearch()
    total_trans = 0
    for i in range(0, len(shard_dirs), shards_per_flush):
        group = shard_dirs[i : i + shards_per_flush]
        states = [s for sd in group for s in _load_shard(sd)]
        transitions, beam_dead = label_placement_states(
            searcher, states, 16, 200, m.max_len, progress=progress
        )
        if transitions:
            existing, _ = _load_existing(dataset_path, is_placement=True)
            _merge_and_save(transitions, existing, dataset_path, is_placement=True)
            total_trans += len(transitions)
        for sd in group:
            shutil.rmtree(sd)  # durably labeled -> consume so a re-run skips it
        remaining = len(shard_dirs) - i - len(group)
        print(
            f"Merged {len(group)} shard(s) | +{len(transitions)} transitions "
            f"(beam_dead {beam_dead}) | {total_trans} total | {remaining} shard(s) left",
            flush=True,
        )
    print(f"Done: {total_trans} transitions; all shards consumed.", flush=True)
    return 0


def main(cli_args):
    from types import SimpleNamespace
    from qtris.config import ModelConfig, EnvConfig

    m = ModelConfig()
    e = EnvConfig()
    args = SimpleNamespace(
        family=cli_args.family,
        policy_checkpoint=getattr(cli_args, "checkpoint", None),
        dataset_path=getattr(cli_args, "output", None),
        num_steps=cli_args.num_steps,
        seed=getattr(cli_args, "seed", 10_000_000),
        search_depth=16,
        beam_width=200,
        queue_size=m.queue_size,
        max_len=m.max_len,
        key_dim=m.key_dim,
        piece_dim=m.piece_dim,
        depth=m.depth,
        num_heads=m.num_heads,
        num_layers=m.num_layers,
        dropout_rate=m.dropout_rate,
        max_holes=e.max_holes,
        max_steps_env=9_999_999,
        garbage_chance=e.garbage_chance,
        garbage_min=e.garbage_min,
        garbage_max=e.garbage_max,
        garbage_push_delay=e.garbage_push_delay,
        num_row_tiers=m.num_row_tiers,
        headless=getattr(cli_args, "headless", False),
        log_every=1000,
    )

    family_defaults = {
        "ar": {
            "policy_checkpoint": "checkpoints/ar_pretrained_policy",
            "dataset_path": "datasets/tetris_expert_dataset_b2b",
            "build_model": _build_ar_model,
        },
        "flat": {
            "policy_checkpoint": "checkpoints/flat_pretrained_policy",
            "dataset_path": "datasets/tetris_expert_dataset_flat",
            "build_model": _build_flat_model,
        },
        "placement": {
            "policy_checkpoint": "checkpoints/placement_pretrained_policy",
            "dataset_path": "datasets/tetris_oracle_placement",
            "build_model": _build_placement_model,
        },
    }
    cfg = family_defaults[args.family]
    is_placement = args.family == "placement"
    policy_checkpoint = str(args.policy_checkpoint or cfg["policy_checkpoint"])
    dataset_path = str(args.dataset_path) if args.dataset_path else cfg["dataset_path"]

    p_model = cfg["build_model"](args)

    p_checkpoint = tf.train.Checkpoint(model=p_model)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint,
        policy_checkpoint,
        max_to_keep=3,
    )
    if p_checkpoint_manager.latest_checkpoint is None:
        print(f"ERROR: no policy checkpoint at {policy_checkpoint}.", flush=True)
        return 1
    p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
    print(
        f"Restored {args.family} policy from {p_checkpoint_manager.latest_checkpoint}",
        flush=True,
    )

    existing, existing_count = _load_existing(dataset_path, is_placement)

    print(
        f"Collecting {args.num_steps} DAgger steps "
        f"(seed offset {args.seed + existing_count})...",
        flush=True,
    )

    collect_fn = collect_dagger_placement if is_placement else collect_dagger
    new_transitions, beam_dead, deaths, max_b2b, policy_disagrees, app = collect_fn(
        p_model=p_model,
        seed=args.seed + existing_count,
        num_steps=args.num_steps,
        search_depth=args.search_depth,
        beam_width=args.beam_width,
        queue_size=args.queue_size,
        max_len=args.max_len,
        max_holes=args.max_holes,
        max_steps_env=args.max_steps_env,
        garbage_chance=args.garbage_chance,
        garbage_min=args.garbage_min,
        garbage_max=args.garbage_max,
        garbage_push_delay=args.garbage_push_delay,
        num_row_tiers=args.num_row_tiers,
        headless=args.headless,
        log_every=args.log_every,
    )

    if not new_transitions:
        print("No new transitions collected; dataset unchanged.", flush=True)
        return 0

    print(
        f"Collected {len(new_transitions)} DAgger transitions | "
        f"beam_dead: {beam_dead} | deaths: {deaths} | max_b2b: {max_b2b} | "
        f"APP: {app:.3f} | policy≠beam: {policy_disagrees}",
        flush=True,
    )

    _merge_and_save(new_transitions, existing, dataset_path, is_placement)
    return 0
