"""Single-player AlphaZero self-improvement loop for the placement model.

Same solo-play env + random-garbage settings as the PPO trainer
(`qtris.training.placement`), but PPO is replaced by an AlphaZero loop: PUCT MCTS
(`PlacementMCTS`) plays self-play games using the net's policy as priors and its value
head at leaves; the net is then trained to imitate the search (policy -> root visit
distribution) and to regress the search-bootstrapped self-play return (value -> the
discounted attack/death return the search optimizes, with the net's own root value
bootstrapping the truncation horizon). Both the policy and the value improve via search.
Positions are accumulated in a multi-generation replay buffer (decorrelates the tiny
on-policy batches and resists drift off the pretrained init). The value lives in
`return_scale` units - the same units the MCTS leaf bootstrap + edge rewards use, so the
search value/reward weighting is consistent. wandb observability via `qtris.observability`.
"""

from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.observability.models import AlphaZeroTrainConfig, SingleAgentAZLog
from qtris.observability.wandb_backend import finish, init_run, log_step
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
        policy_loss = tf.reduce_mean(
            -tf.reduce_sum(batch["pi_target"] * log_probs, axis=-1)
        )
        value_loss = tf.reduce_mean((values[:, 0] - batch["value_target"]) ** 2)
        loss = policy_loss + value_coef * value_loss

    grads = tape.gradient(loss, net.trainable_variables)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))

    probs = tf.nn.softmax(masked, axis=-1)
    entropy = tf.reduce_mean(-tf.reduce_sum(probs * log_probs, axis=-1))
    ret_var = tf.math.reduce_variance(batch["value_target"])
    res_var = tf.math.reduce_variance(batch["value_target"] - values[:, 0])
    explained_var = 1.0 - tf.math.divide_no_nan(res_var, ret_var)
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "explained_var": explained_var,
        "value_mean": tf.reduce_mean(values[:, 0]),
    }


def _build_envs(num_games, queue_size, max_holes, max_height, max_steps, max_len, args):
    gmin = getattr(args, "garbage_chance_min", 0.0)
    gmax = getattr(args, "garbage_chance_max", 0.2)
    chances = [
        gmin + (gmax - gmin) * i / max(num_games - 1, 1) for i in range(num_games)
    ]
    return [
        PyTetrisEnv(
            queue_size=queue_size,
            max_holes=max_holes,
            max_height=max_height,
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
        )
        for i in range(num_games)
    ]


def _estimate_return_var(mcts, envs, searcher, forced_drop, gamma, horizon, num_envs):
    """Pre-training estimate of the discounted-return variance to seed return_scale.
    The EMA on return_var is slow (0.01), so a fresh warm-start would sit mis-scaled for many
    generations - giving the value head a huge initial loss and badly-scaled targets. One short
    self-play rollout with the warm-started net, over the same attack-only return the value head
    regresses (pure MC, no bootstrap), gives a calibrated starting scale."""
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
            if died or envs[i]._episode_ended:
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
    mini_batch_size = getattr(args, "mini_batch_size", 256)
    num_epochs = getattr(args, "num_epochs", 2)
    value_coef = getattr(args, "value_coef", 1.0)
    learning_rate = getattr(args, "learning_rate", 1e-4)
    replay_capacity = getattr(args, "replay_capacity", 25_000)
    gae_lambda = getattr(args, "gae_lambda", 1.0)

    cfg = MCTSConfig(
        num_simulations=getattr(args, "num_simulations", 64),
        c_puct=getattr(args, "c_puct", 1.5),
        dirichlet_alpha=getattr(args, "dirichlet_alpha", 0.3),
        dirichlet_eps=getattr(args, "dirichlet_eps", 0.25),
        gamma=getattr(args, "gamma", GAMMA),
        temp_moves=getattr(args, "temp_moves", 12),
        w_attack=getattr(args, "w_attack", 1.0),
        w_b2b=getattr(args, "w_b2b", 1.0),
        w_death=getattr(args, "w_death", 100.0),
        leaves_per_round=getattr(args, "leaves_per_round", 4),
        vloss=getattr(args, "vloss", 1.0),
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
        w_b2b=cfg.w_b2b,
        w_death=cfg.w_death,
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        value_coef=value_coef,
        learning_rate=learning_rate,
        replay_capacity=replay_capacity,
        gae_lambda=gae_lambda,
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
    manager = tf.train.CheckpointManager(
        checkpoint, "checkpoints/placement_az", max_to_keep=3
    )
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
        max_height=18,
        max_steps=1024,
        max_len=max_len,
        args=args,
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

    wandb_run = init_run(project="Tetris", config=config)
    return_var = float(return_scale) ** 2

    # Seed return_scale from a warm-start rollout (skip when resuming a calibrated AZ ckpt,
    # whose return_scale was restored above). Avoids the slow-EMA cold-start mis-scaling.
    if not manager.latest_checkpoint:
        return_var = _estimate_return_var(
            mcts, envs, searcher, forced_drop, cfg.gamma, horizon, num_games
        )
        return_scale.assign(tf.sqrt(return_var))
        print(
            f"Seeded return_scale={float(return_scale):.3f} from warm-start.",
            flush=True,
        )
        for env in envs:
            env._reset()
        move_count = np.zeros(num_games, dtype=np.int64)

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

        scale = float(np.sqrt(return_var))
        for t in range(horizon):
            temps = np.where(move_count < cfg.temp_moves, 1.0, 0.0).astype(np.float32)
            results = mcts.search(envs, scale, temps)
            for i, res in enumerate(results):
                if res["dead"]:
                    ts = envs[i]._step(forced_drop.copy())
                    # Dead root = death: same attack-only reward the search optimizes, minus the
                    # death penalty (b2b is a search-time leaf bootstrap, not a realized reward).
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
                # attack-only return target (see above), minus the death penalty on a fatal move
                rewards[t, i] = cfg.w_attack * attack - (cfg.w_death if died else 0.0)
                attacks[t, i] = attack
                clears[t, i] = clear
                if died or envs[i]._episode_ended:
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

        # Value target = the discounted self-play return the search optimizes (attack - death),
        # bootstrapped at the horizon by the net's own root value (root_values, no simulation),
        # then put in return_scale units to match the MCTS leaf bootstrap + edge rewards. lam=1
        # gives the MC return + a single boundary bootstrap; lam<1 trades variance for value bias.
        last_v = mcts.root_values(envs)
        _adv, returns = compute_gae_and_returns(
            v_root[..., None],
            last_v[..., None],
            rewards[..., None],
            dones[..., None],
            cfg.gamma,
            gae_lambda,
            horizon,
            num_games,
        )
        value_tgt = returns.numpy()[..., 0] / (scale + 1e-8)

        # EMA the raw-return variance for the next generation's return_scale (search normalizer +
        # value-target units). Updated before the skip checks so the scale evolves every gen.
        raw_returns = compute_raw_returns(
            rewards[..., None], dones[..., None], cfg.gamma, horizon, num_games
        )
        return_var = 0.99 * return_var + 0.01 * float(
            tf.math.reduce_variance(raw_returns)
        )
        return_scale.assign(tf.sqrt(return_var))

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

        updates = 0
        step_out = None
        for batch in ds:
            step_out = train_step(net, batch, tf.constant(value_coef, tf.float32))
            updates += 1
        if step_out is None:
            print(f"Gen {gen}: no batch produced; skipping update.", flush=True)
            continue

        deaths_per_game = dones.sum(axis=0)
        b2b_series = bcg[..., 0][storable]
        combo_series = bcg[..., 1][storable]
        if gen % 1 == 0:
            log_step(
                SingleAgentAZLog(
                    policy_loss=step_out["policy_loss"],
                    value_loss=step_out["value_loss"],
                    entropy=step_out["entropy"],
                    explained_var=step_out["explained_var"],
                    value_mean=step_out["value_mean"],
                    return_var=return_var,
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
                f"Reward: {float(rewards.sum(axis=0).mean()):3.1f} | "
                f"Deaths: {float(deaths_per_game.mean()):1.2f} | Updates: {updates}",
                flush=True,
            )
        if gen % 5 == 0:
            manager.save()

    finish(wandb_run)
