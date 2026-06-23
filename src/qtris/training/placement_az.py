"""Single-player AlphaZero self-improvement loop for the placement model.

Same solo-play env + random-garbage settings as the PPO trainer
(`qtris.training.placement`), but PPO is replaced by an AlphaZero loop: PUCT MCTS
(`PlacementMCTS`) plays self-play games using the net's policy as priors and its value
head at leaves; the net is then trained to imitate the search (policy -> root visit
distribution) and to regress the search-bootstrapped self-play return (attack minus the
death penalty, no shaping). Both the policy and the value improve via search. Positions
accumulate in a multi-generation replay buffer (decorrelates the tiny on-policy batches and
resists drift off the pretrained init). The value lives in `return_scale` units - the same
units the MCTS edges use, so the search value/reward weighting is consistent. `policy_kl`
(KL of the net to the visit targets) and `update_kl` (how far the policy moves per
generation of updates) are logged to watch optimization divergence. Observability via
`qtris.observability`.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.observability.models import AlphaZeroTrainConfig, SingleAgentAZLog
from qtris.observability.backend import finish, init_run, log_step
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.search.placement_search import placement_step
from qtris.training.gae import compute_gae_and_returns, compute_raw_returns

GAMMA = 0.99


def _flat(arr, sel):
    """Collapse the (horizon, num_games) leading axes and keep storable rows."""
    return arr.reshape((-1,) + arr.shape[2:])[sel]


@tf.function
def train_step(net, batch, value_coef):
    cand_mask = batch["cand_mask"]
    # Optional per-position policy weight (1=train policy here, 0=value-only). Absent for
    # solo AZ, so the policy terms reduce to a plain mean and behavior is unchanged.
    pm = batch.get("policy_mask")
    with tf.GradientTape() as tape:
        logits, values = net(
            (
                batch["boards"],
                batch["pieces"],
                batch["bcg"],
                batch["cand_placements"],
                cand_mask,
            ),
            training=True,
        )
        masked = tf.where(cand_mask, logits, tf.constant(-1e9, tf.float32))
        log_probs = tf.nn.log_softmax(masked, axis=-1)
        ce = -tf.reduce_sum(batch["pi_target"] * log_probs, axis=-1)
        if pm is not None:
            pnorm = tf.reduce_sum(pm) + 1e-8
            policy_loss = tf.reduce_sum(pm * ce) / pnorm
        else:
            policy_loss = tf.reduce_mean(ce)
        value_loss = tf.reduce_mean((values[:, 0] - batch["value_target"]) ** 2)
        loss = policy_loss + value_coef * value_loss

    grads = tape.gradient(loss, net.trainable_variables)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))

    probs = tf.nn.softmax(masked, axis=-1)
    ent = -tf.reduce_sum(probs * log_probs, axis=-1)
    # Exact KL(pi_target || p_net) = CE - H(pi_target); zero target arms contribute 0.
    tgt = batch["pi_target"]
    tgt_ent = -tf.reduce_sum(tgt * tf.math.log(tgt + 1e-12), axis=-1)
    if pm is not None:
        entropy = tf.reduce_sum(pm * ent) / pnorm
        tgt_entropy = tf.reduce_sum(pm * tgt_ent) / pnorm
    else:
        entropy = tf.reduce_mean(ent)
        tgt_entropy = tf.reduce_mean(tgt_ent)
    ret_var = tf.math.reduce_variance(batch["value_target"])
    res_var = tf.math.reduce_variance(batch["value_target"] - values[:, 0])
    explained_var = 1.0 - tf.math.divide_no_nan(res_var, ret_var)
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "policy_kl": policy_loss - tgt_entropy,
        "explained_var": explained_var,
        "value_mean": tf.reduce_mean(values[:, 0]),
    }


@tf.function
def _gen_log_probs(net, boards, pieces, bcg, cand_pl, cand_mk):
    """Masked policy log-probs over one generation's states (fixed horizon*num_games
    shape, so this traces once). Used to measure update_kl: how far the policy moved
    over the generation's update steps."""
    logits, _ = net((boards, pieces, bcg, cand_pl, cand_mk), training=False)
    masked = tf.where(cand_mk, logits, tf.constant(-1e9, tf.float32))
    return tf.nn.log_softmax(masked, axis=-1)


def _load_trace_pools(traces_dir):
    """Load the trace library: tier subdirs of .npy attack streams, sorted name =
    difficulty order (e.g. 00_sims16 .. 03_sims256, 99_recent). Skips empty tiers."""
    pools = {}
    for tier in sorted(p for p in Path(traces_dir).iterdir() if p.is_dir()):
        traces = [np.load(f) for f in sorted(tier.glob("*.npy"))]
        traces = [t for t in traces if t.size > 0]
        if traces:
            pools[tier.name] = traces
    return pools


def _trace_tier_map(num_games, trace_free_envs, tiers):
    """env index -> tier name (None = garbage-free). First trace_free_envs envs are
    free; the rest split evenly across tiers, weakest (first sorted name) first."""
    mapping = {}
    rest = num_games - trace_free_envs
    for i in range(num_games):
        if i < trace_free_envs or not tiers:
            mapping[i] = None
        else:
            mapping[i] = tiers[(i - trace_free_envs) * len(tiers) // rest]
    return mapping


# Difficulty-curriculum controller: keep per-game deaths inside [LO, HI] by ramping a
# continuous difficulty index. Asymmetric (back off faster than ramp up) so a hard generation
# is corrected before the spiral that weak search + over-hard garbage produces (the collapse
# mechanism: difficulty above what competence + sims can survive).
CUR_DEATH_LO, CUR_DEATH_HI = 0.4, 1.0
CUR_STEP_UP, CUR_STEP_DOWN = 0.15, 0.40


def _curriculum_tier_map(num_games, trace_free_envs, tiers, difficulty):
    """env index -> tier name under the feedback curriculum. Trace envs face a spread of tiers
    from the weakest up to the current difficulty ceiling, so the policy always keeps some
    survivable envs to learn from plus harder ones to push competence. difficulty is a
    continuous index in [0, len(tiers)-1]."""
    mapping = {}
    n_trace = max(num_games - trace_free_envs, 1)
    top = max(len(tiers) - 1, 0)
    for i in range(num_games):
        if i < trace_free_envs or not tiers:
            mapping[i] = None
        else:
            j = i - trace_free_envs
            idx = round(difficulty * j / max(n_trace - 1, 1))
            mapping[i] = tiers[min(max(idx, 0), top)]
    return mapping


def _build_envs(
    num_games,
    queue_size,
    max_holes,
    max_steps,
    max_len,
    args,
    trace_pools=None,
):
    gmin = getattr(args, "garbage_chance_min", 0.0)
    gmax = getattr(args, "garbage_chance_max", 0.2)
    chances = [
        gmin + (gmax - gmin) * i / max(num_games - 1, 1) for i in range(num_games)
    ]
    tier_map = {}
    if trace_pools:
        tier_map = _trace_tier_map(
            num_games, getattr(args, "trace_free_envs", 2), list(trace_pools)
        )
        chances = [0.0] * num_games  # pressure comes from traces (or nothing)
        print(
            "Trace garbage tiers: "
            + ", ".join(f"{t}({len(p)})" for t, p in trace_pools.items())
            + " | env map: "
            + " ".join(str(tier_map[i] or "-") for i in range(num_games)),
            flush=True,
        )
    return [
        PyTetrisEnv(
            queue_size=queue_size,
            max_holes=max_holes,
            max_steps=max_steps,
            max_len=max_len,
            pathfinding=False,
            seed=123 + i,
            idx=i,
            garbage_chance=chances[i],
            garbage_min=getattr(args, "garbage_rows_min", 1),
            garbage_max=getattr(args, "garbage_rows_max", 4),
            num_row_tiers=2,
            placement_candidates=False,
            garbage_traces=trace_pools.get(tier_map[i]) if tier_map.get(i) else None,
        )
        for i in range(num_games)
    ]


def _estimate_return_var(mcts, envs, searcher, forced_drop, gamma, horizon, num_envs):
    """Pre-training estimate of the discounted-return variance to seed return_scale, which is
    then FROZEN for the whole run. One short self-play rollout with the warm-started net, over
    the same attack-only return the value head regresses (pure MC, no bootstrap), gives a
    calibrated scale; without it a fresh start would sit at 1.0 with wildly mis-scaled targets."""
    rewards = np.zeros((horizon, num_envs), dtype=np.float32)
    dones = np.zeros((horizon, num_envs), dtype=np.float32)
    for env in envs:
        env._reset()
    for t in range(horizon):
        results = mcts.search(envs, 1.0, np.ones(num_envs, dtype=np.float32))
        for i, res in enumerate(results):
            if res["dead"]:
                envs[i]._step(forced_drop.copy())
                rewards[t, i] = -mcts.cfg.w_death
                dones[t, i] = 1.0
                envs[i]._reset()
                continue
            _total, attack, _clear, died = placement_step(
                envs[i], searcher, res["descriptor"]
            )
            rewards[t, i] = mcts.cfg.w_attack * attack - (
                mcts.cfg.w_death if died else 0.0
            )
            terminal = died or envs[i]._episode_ended
            if terminal:
                dones[t, i] = 1.0
                envs[i]._reset()
    raw = compute_raw_returns(
        rewards[..., None], dones[..., None], gamma, horizon, num_envs
    )
    return max(float(tf.math.reduce_variance(raw)), 1.0)


def main(args):
    piece_dim, depth, num_heads, num_layers = 8, 64, 4, 4
    queue_size, max_len = 5, 15
    num_games = getattr(args, "num_games", 16)
    horizon = getattr(args, "horizon", 32)
    num_generations = getattr(args, "num_generations", 1_000_000)
    mini_batch_size = getattr(args, "batch_size", 256)
    num_epochs = getattr(args, "num_epochs", 2)
    value_coef = getattr(args, "value_coef", 1.0)
    learning_rate = getattr(args, "learning_rate", 1e-4)
    replay_capacity = getattr(args, "replay_capacity", 25_000)
    gae_lambda = getattr(args, "gae_lambda", 1.0)
    garbage_traces = getattr(args, "garbage_traces", None)
    trace_free_envs = getattr(args, "trace_free_envs", 2)
    trace_harvest_cap = getattr(args, "trace_harvest_cap", 256)
    return_scale_override = getattr(args, "return_scale", None)
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints/placement_az")
    run_name = getattr(args, "run_name", None)
    no_harvest = getattr(args, "no_harvest", False)
    trace_tiers = getattr(args, "trace_tiers", None)
    seed = getattr(args, "seed", None)
    curriculum = getattr(args, "curriculum", False)
    cur_d = float(getattr(args, "curriculum_start", 0.0))

    if seed is not None:
        np.random.seed(seed)

    trace_pools = _load_trace_pools(garbage_traces) if garbage_traces else None
    if trace_pools is not None and trace_tiers:
        keep = trace_tiers.split(",")
        missing = [t for t in keep if t not in trace_pools]
        if missing:
            raise SystemExit(f"--trace-tiers: unknown tiers {missing}.")
        trace_pools = {t: trace_pools[t] for t in sorted(keep)}

    cfg = MCTSConfig(
        num_simulations=getattr(args, "num_simulations", 64),
        c_puct=getattr(args, "c_puct", 1.5),
        dirichlet_alpha=getattr(args, "dirichlet_alpha", 0.3),
        dirichlet_eps=getattr(args, "dirichlet_eps", 0.25),
        gamma=getattr(args, "gamma", GAMMA),
        temp_moves=getattr(args, "temp_moves", 12),
        w_attack=getattr(args, "w_attack", 1.0),
        w_death=getattr(args, "w_death", 100.0),
        leaves_per_round=getattr(args, "leaves_per_round", 4),
        vloss=getattr(args, "vloss", 1.0),
    )

    net = PlacementPolicyValueNet(
        batch_size=num_games,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=0.0,
    )
    optimizer = keras.optimizers.Adam(learning_rate, clipnorm=0.5)
    net.compile(optimizer=optimizer, jit_compile=True)
    net(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    net.summary()

    return_scale = tf.Variable(
        1.0, trainable=False, dtype=tf.float32, name="return_scale"
    )
    checkpoint = tf.train.Checkpoint(
        model=net,
        optimizer=optimizer,
        return_scale=return_scale,
    )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Resumed AZ checkpoint {manager.latest_checkpoint}.", flush=True)
    else:
        warm = tf.train.latest_checkpoint("checkpoints/placement_pretrained_policy")
        if warm is not None:
            # Warm-start the policy only. The value head was pretrained against the oracle
            # depth-0 max (value_scale units); it retargets to the search return (return_scale
            # units) over the first few generations - a transient that self-corrects.
            tf.train.Checkpoint(model=net).restore(warm).expect_partial()
            print(f"Warm-started policy from BC checkpoint {warm}.", flush=True)

    envs = _build_envs(
        num_games,
        queue_size,
        max_holes=50,
        max_steps=None,  # a cap counts truncations as deaths and cuts the bootstrap
        max_len=max_len,
        args=args,
        trace_pools=trace_pools,
    )
    mcts = PlacementMCTS(net, cfg)
    searcher = (
        CB2BSearch()
    )  # lock-score core for committing the chosen move by descriptor

    forced_drop = np.full(max_len, Keys.PAD, dtype=np.int64)
    forced_drop[0], forced_drop[1] = Keys.START, Keys.HARD_DROP

    for env in envs:
        env._reset()
    move_count = np.zeros(num_games, dtype=np.int64)

    # Seed return_scale from a warm-start rollout (skip when resuming a calibrated AZ ckpt,
    # whose return_scale was restored above, or when --return-scale forces it), then FROZEN:
    # AZ normalizes nothing (bounded z target) and MuZero min-max normalizes Q in-tree; a
    # running return-variance EMA is a PPO-style trick, and its loosening (variance up ->
    # scale up -> death penalty down) amplified every collapse.
    resumed = manager.latest_checkpoint is not None
    if not resumed:
        if return_scale_override is not None:
            return_scale.assign(return_scale_override)
            print(
                f"Forced return_scale={float(return_scale):.3f} (frozen).", flush=True
            )
        else:
            return_scale.assign(
                tf.sqrt(
                    _estimate_return_var(
                        mcts, envs, searcher, forced_drop, cfg.gamma, horizon, num_games
                    )
                )
            )
            print(
                f"Seeded return_scale={float(return_scale):.3f} from warm-start (frozen).",
                flush=True,
            )
            for env in envs:
                env._reset()
            move_count = np.zeros(num_games, dtype=np.int64)
    elif return_scale_override is not None:
        print(
            f"--return-scale {return_scale_override} ignored: resumed ckpt keeps "
            f"return_scale={float(return_scale):.3f}.",
            flush=True,
        )

    config = AlphaZeroTrainConfig(
        num_games=num_games,
        horizon=horizon,
        num_simulations=cfg.num_simulations,
        c_puct=cfg.c_puct,
        gamma=cfg.gamma,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        temp_moves=cfg.temp_moves,
        w_attack=cfg.w_attack,
        w_death=cfg.w_death,
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        value_coef=value_coef,
        learning_rate=learning_rate,
        replay_capacity=replay_capacity,
        gae_lambda=gae_lambda,
        garbage_traces=garbage_traces,
        trace_free_envs=trace_free_envs,
        return_scale=float(return_scale),
        resumed=resumed,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        harvest=not no_harvest,
        trace_tiers=trace_tiers,
        seed=seed,
        curriculum=curriculum,
        curriculum_start=cur_d,
    )
    run = init_run(
        project="Tetris",
        config=config,
        wandb_mirror=getattr(args, "wandb", False),
        run_name=run_name,
    )

    # Difficulty curriculum: start the trace envs at the configured difficulty floor (the fixed
    # spread in _build_envs is the d=full assignment); the per-gen controller ramps from here.
    if curriculum and trace_pools:
        cmap = _curriculum_tier_map(
            num_games, trace_free_envs, list(trace_pools), cur_d
        )
        for i, e in enumerate(envs):
            e._garbage_traces = trace_pools.get(cmap[i]) if cmap.get(i) else None

    # Multi-generation replay of storable positions (decorrelates the tiny on-policy batches
    # and resists drift off the pretrained init). Each entry is one generation's numpy arrays;
    # oldest generations are evicted once the total position count exceeds replay_capacity.
    replay = deque()
    replay_size = 0

    for gen in range(num_generations):
        boards = np.zeros((horizon, num_games, 24, 10, 1), dtype=np.float32)
        pieces = np.zeros((horizon, num_games, queue_size + 2), dtype=np.int64)
        bcg = np.zeros((horizon, num_games, 3), dtype=np.float32)
        cand_pl = np.zeros(
            (horizon, num_games, CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM),
            dtype=np.float32,
        )
        cand_mk = np.zeros((horizon, num_games, CANDIDATE_CAPACITY), dtype=bool)
        pi_tgt = np.zeros((horizon, num_games, CANDIDATE_CAPACITY), dtype=np.float32)
        v_root = np.zeros(
            (horizon, num_games), dtype=np.float32
        )  # net root value baseline
        rewards = np.zeros((horizon, num_games), dtype=np.float32)
        attacks = np.zeros((horizon, num_games), dtype=np.float32)
        clears = np.zeros((horizon, num_games), dtype=np.float32)
        dones = np.zeros((horizon, num_games), dtype=np.float32)
        storable = np.zeros((horizon, num_games), dtype=bool)
        visits = np.zeros((horizon, num_games), dtype=np.float32)

        garb_before = [
            (e._garbage_spawned_rows, e._garbage_spawned_events, e._garbage_pushed_rows)
            for e in envs
        ]

        scale = float(return_scale)
        for t in range(horizon):
            temps = np.where(move_count < cfg.temp_moves, 1.0, 0.0).astype(np.float32)
            results = mcts.search(envs, scale, temps)
            for i, res in enumerate(results):
                if res["dead"]:
                    ts = envs[i]._step(forced_drop.copy())
                    # Dead root = death: attack-only reward minus the death penalty.
                    rewards[t, i] = (
                        cfg.w_attack * float(ts.reward["attack"]) - cfg.w_death
                    )
                    attacks[t, i] = float(ts.reward["attack"])
                    clears[t, i] = float(ts.reward["clear"])
                    dones[t, i] = 1.0
                    envs[i]._reset()
                    move_count[i] = 0
                    continue
                boards[t, i] = res["board"]
                pieces[t, i] = res["pieces"]
                bcg[t, i] = res["bcg"]
                cand_pl[t, i] = res["cand_placements"]
                cand_mk[t, i] = res["cand_mask"]
                pi_tgt[t, i] = res["pi"]
                v_root[t, i] = res[
                    "value"
                ]  # net value of this root (return-bootstrap baseline)
                visits[t, i] = res["visits"]
                storable[t, i] = True
                _total, attack, clear, died = placement_step(
                    envs[i], searcher, res["descriptor"]
                )
                # attack-only realized reward, minus the death penalty on a fatal move
                rewards[t, i] = cfg.w_attack * attack - (cfg.w_death if died else 0.0)
                attacks[t, i] = attack
                clears[t, i] = clear
                terminal = died or envs[i]._episode_ended
                if terminal:
                    dones[t, i] = 1.0
                    envs[i]._reset()
                    move_count[i] = 0
                else:
                    move_count[i] += 1

        sel = storable.reshape(-1)
        n_new = int(sel.sum())
        if n_new == 0:
            print(
                f"Gen {gen}: no storable samples (all dead); skipping update.",
                flush=True,
            )
            continue

        # Value target = the discounted self-play return, bootstrapped at the horizon by the
        # net's own root value (root_values, no simulation), then put in return_scale units to
        # match the MCTS edges. lam=1 gives the MC return + a single boundary bootstrap; lam<1
        # trades variance for value bias. Net values are in return_scale units, rewards are raw:
        # scale values up before mixing.
        last_v = mcts.root_values(envs) * scale
        _adv, returns = compute_gae_and_returns(
            (v_root * scale)[..., None],
            last_v[..., None],
            rewards[..., None],
            dones[..., None],
            cfg.gamma,
            gae_lambda,
            horizon,
            num_games,
        )
        value_tgt = returns.numpy()[..., 0] / (scale + 1e-8)

        # Measured per-gen return variance: DIAGNOSTIC ONLY (a rise preceding a reward fall
        # is the collapse fingerprint). return_scale stays frozen at its seed.
        raw_returns = compute_raw_returns(
            rewards[..., None], dones[..., None], cfg.gamma, horizon, num_games
        )
        gen_return_var = float(tf.math.reduce_variance(raw_returns))

        # Append this generation's storable positions to the replay buffer; evict oldest gens
        # once total positions exceed replay_capacity.
        replay.append(
            {
                "boards": _flat(boards, sel),
                "pieces": _flat(pieces, sel),
                "bcg": _flat(bcg, sel),
                "cand_placements": _flat(cand_pl, sel),
                "cand_mask": _flat(cand_mk, sel),
                "pi_target": _flat(pi_tgt, sel),
                "value_target": _flat(value_tgt, sel),
            }
        )
        replay_size += n_new
        while replay_size > replay_capacity and len(replay) > 1:
            replay_size -= len(replay.popleft()["value_target"])

        if replay_size < mini_batch_size:
            print(
                f"Gen {gen}: replay {replay_size} < batch {mini_batch_size}; skipping update.",
                flush=True,
            )
            continue

        # Train on minibatches sampled from the whole replay buffer (old + new mixed). Step count
        # scales with the NEW data so per-gen compute stays bounded as the buffer grows.
        full = {k: np.concatenate([e[k] for e in replay], axis=0) for k in replay[0]}
        total_steps = num_epochs * max(1, n_new // mini_batch_size)
        ds = (
            tf.data.Dataset.from_tensor_slices(full)
            .shuffle(replay_size)
            .repeat()
            .batch(mini_batch_size, drop_remainder=True)
            .take(total_steps)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Snapshot the policy on this generation's states (full fixed-shape arrays, one
        # trace) so update_kl below measures how far the update steps moved it.
        gen_inputs = (
            tf.constant(boards.reshape(-1, 24, 10, 1)),
            tf.constant(pieces.reshape(-1, queue_size + 2)),
            tf.constant(bcg.reshape(-1, 3)),
            tf.constant(cand_pl.reshape(-1, CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM)),
            tf.constant(cand_mk.reshape(-1, CANDIDATE_CAPACITY)),
        )
        lp_before = _gen_log_probs(net, *gen_inputs).numpy()[sel]

        updates = 0
        step_out = None
        for batch in ds:
            step_out = train_step(net, batch, tf.constant(value_coef, tf.float32))
            updates += 1
        if step_out is None:
            print(f"Gen {gen}: no batch produced; skipping update.", flush=True)
            continue

        # Per-generation policy divergence: mean KL(p_before || p_after) over the gen's
        # storable states. Illegal arms sit at the same -1e9 logit on both sides, so they
        # contribute ~0 mass and ~0 log-ratio.
        lp_after = _gen_log_probs(net, *gen_inputs).numpy()[sel]
        update_kl = float(
            (np.exp(lp_before) * (lp_before - lp_after)).sum(axis=-1).mean()
        )

        deaths_per_game = dones.sum(axis=0)
        b2b_series = bcg[..., 0][storable]
        combo_series = bcg[..., 1][storable]

        # Feedback curriculum: ramp difficulty toward the deaths deadband, reassign env tiers.
        if curriculum and trace_pools:
            ad = float(deaths_per_game.mean())
            if ad < CUR_DEATH_LO:
                cur_d = min(cur_d + CUR_STEP_UP, len(trace_pools) - 1)
            elif ad > CUR_DEATH_HI:
                cur_d = max(cur_d - CUR_STEP_DOWN, 0.0)
            cmap = _curriculum_tier_map(
                num_games, trace_free_envs, list(trace_pools), cur_d
            )
            for i, e in enumerate(envs):
                e._garbage_traces = trace_pools.get(cmap[i]) if cmap.get(i) else None

        # Incoming-garbage telemetry from the env counters (per-gen deltas).
        # cancel_frac is approximate per gen: pending-queue carryover and reset-dropped
        # entries land on the "cancelled" side; accurate in the long run.
        g_rows = sum(e._garbage_spawned_rows for e in envs) - sum(
            b[0] for b in garb_before
        )
        g_events = sum(e._garbage_spawned_events for e in envs) - sum(
            b[1] for b in garb_before
        )
        g_pushed = sum(e._garbage_pushed_rows for e in envs) - sum(
            b[2] for b in garb_before
        )
        garbage_in_max = max(e._garbage_max_event for e in envs)
        for e in envs:
            e._garbage_max_event = 0
        n_steps = horizon * num_games
        if gen % 1 == 0:
            log_step(
                SingleAgentAZLog(
                    policy_loss=step_out["policy_loss"],
                    value_loss=step_out["value_loss"],
                    entropy=step_out["entropy"],
                    policy_kl=step_out["policy_kl"],
                    update_kl=update_kl,
                    explained_var=step_out["explained_var"],
                    value_mean=step_out["value_mean"],
                    return_var=gen_return_var,
                    return_scale=scale,
                    avg_total_reward=float(rewards.sum(axis=0).mean()),
                    avg_attacks=float(attacks.sum(axis=0).mean()),
                    avg_clears=float(clears.sum(axis=0).mean()),
                    avg_deaths=float(deaths_per_game.mean()),
                    avg_pieces=float((horizon / (deaths_per_game + 1)).mean()),
                    avg_b2b=float(b2b_series.mean()) if b2b_series.size else 0.0,
                    max_b2b=float(b2b_series.max()) if b2b_series.size else 0.0,
                    avg_combo=float(combo_series.mean()) if combo_series.size else 0.0,
                    surge_rate=float((b2b_series >= 4).mean())
                    if b2b_series.size
                    else 0.0,
                    garbage_in_app=g_rows / n_steps,
                    garbage_in_rate=g_events / n_steps,
                    garbage_in_chunk=g_rows / g_events if g_events else 0.0,
                    garbage_in_max=float(garbage_in_max),
                    garbage_cancel_frac=1.0 - g_pushed / g_rows if g_rows else 0.0,
                    trace_pool_size=sum(len(p) for p in trace_pools.values())
                    if trace_pools
                    else 0,
                    curriculum_d=cur_d if curriculum else 0.0,
                    avg_visits=float(visits[storable].mean())
                    if storable.any()
                    else 0.0,
                    dead_rate=float((~storable).mean()),
                    updates=updates,
                    buffer_size=replay_size,
                    board=boards[storable][0, ..., 0],
                )
            )
            print(
                f"Gen {gen} | Policy: {float(step_out['policy_loss']):2.3f} | "
                f"Value: {float(step_out['value_loss']):2.3f} | "
                f"Ent: {float(step_out['entropy']):1.3f} | "
                f"KL: {update_kl:1.4f} | "
                f"Reward: {float(rewards.sum(axis=0).mean()):3.1f} | "
                f"Deaths: {float(deaths_per_game.mean()):1.2f} | Updates: {updates}",
                flush=True,
            )
        if trace_pools is not None and not no_harvest:
            # Rolling harvest: this gen's attack streams become future opponents.
            # Timestamp prefix keeps eviction chronological across resumes.
            recent = Path(garbage_traces) / "99_recent"
            recent.mkdir(exist_ok=True)
            stamp = int(time.time())
            for i in range(num_games):
                if attacks[:, i].any():
                    np.save(recent / f"{stamp}_g{gen:06d}_e{i:02d}.npy", attacks[:, i])
            files = sorted(recent.glob("*.npy"))
            for f in files[: max(0, len(files) - trace_harvest_cap)]:
                f.unlink()
            if gen % 25 == 24 and not curriculum:
                trace_pools = _load_trace_pools(garbage_traces)
                tier_map = _trace_tier_map(
                    num_games, trace_free_envs, list(trace_pools)
                )
                for i, e in enumerate(envs):
                    e._garbage_traces = (
                        trace_pools.get(tier_map[i]) if tier_map.get(i) else None
                    )

        if gen % 5 == 0:
            manager.save()

    finish(run)
