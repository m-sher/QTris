"""Single-player AlphaZero self-improvement loop for the placement model.

Same solo-play env + random-garbage settings as the PPO trainer
(`qtris.training.placement`), but PPO is replaced by an AlphaZero loop: PUCT MCTS
(`PlacementMCTS`) plays self-play games using the net's policy as priors and its value
head at leaves; the net is then trained to imitate the search (policy -> root visit
distribution) and to regress the oracle `evaluate_state` (value -> board eval, the SAME
target pretraining uses, so warm-start from the BC checkpoint doesn't collapse). Iterating
improves the policy via search; the value stays the oracle critic. wandb observability via
`qtris.observability`. No expert/BC anchor - pure self-play.
"""

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
from qtris.training.gae import compute_raw_returns

GAMMA = 0.99


def _flat(arr, sel):
    """Collapse the (horizon, num_games) leading axes and keep storable rows."""
    return arr.reshape((-1,) + arr.shape[2:])[sel]


def _max_cand_value(searcher, env):
    """Value target matching the pretraining target in form: max over the oracle's candidate
    evaluations (full evaluate_state per resulting placement) for this state. Pretraining used
    max(cand_scores)/value_scale from the depth-16 datagen beam; this is the cheap depth-0
    version (one decompose call/move) - keeps the value the oracle critic so warm-start from the
    BC checkpoint doesn't collapse. max over candidates also dodges evaluate_state's near-death
    floor (the best move usually survives)."""
    queue = np.array([p.value for p in env._queue], dtype=np.int32)
    d = searcher.decompose(
        env._board,
        env._active_piece.piece_type.value,
        env._hold_piece.value,
        queue,
        int(env._scorer._b2b),
        int(env._scorer._combo),
        int(env._get_total_garbage()),
        2 * CANDIDATE_CAPACITY,
    )
    return float(d.sum(axis=1).max()) if len(d) else 0.0


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

    cfg = MCTSConfig(
        num_simulations=getattr(args, "num_simulations", 64),
        c_puct=getattr(args, "c_puct", 1.5),
        dirichlet_alpha=getattr(args, "dirichlet_alpha", 0.3),
        dirichlet_eps=getattr(args, "dirichlet_eps", 0.25),
        gamma=getattr(args, "gamma", GAMMA),
        temp_moves=getattr(args, "temp_moves", 12),
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
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        value_coef=value_coef,
        learning_rate=learning_rate,
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
    # Normalizer the value head was pretrained at (value target = max(cand_scores)/value_scale).
    # The AZ value target reuses it so the phi target matches the pretrained value's units.
    value_scale = tf.Variable(
        1.0, trainable=False, dtype=tf.float32, name="value_scale"
    )
    checkpoint = tf.train.Checkpoint(
        model=net,
        optimizer=optimizer,
        return_scale=return_scale,
        value_scale=value_scale,
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
            # Also restore value_scale (the pretrain value normalizer) so the phi value target
            # is in the same units the warm-started value head outputs - no warm-start mismatch.
            tf.train.Checkpoint(model=net, value_scale=value_scale).restore(
                warm
            ).expect_partial()
            print(
                f"Warm-started from BC checkpoint {warm} (value_scale={float(value_scale):.3f}).",
                flush=True,
            )

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
        value_tgt = np.zeros((horizon, num_games), dtype=np.float32)
        rewards = np.zeros((horizon, num_games), dtype=np.float32)
        attacks = np.zeros((horizon, num_games), dtype=np.float32)
        clears = np.zeros((horizon, num_games), dtype=np.float32)
        dones = np.zeros((horizon, num_games), dtype=np.float32)
        storable = np.zeros((horizon, num_games), dtype=bool)
        visits = np.zeros((horizon, num_games), dtype=np.float32)

        scale = float(np.sqrt(return_var))
        vs = float(value_scale) + 1e-8  # value-target normalizer (matches pretraining)
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
                visits[t, i] = res["visits"]
                storable[t, i] = True
                # Value target = max over the oracle's candidate evals for THIS (pre-move) state,
                # /value_scale - the pretraining value target's form (depth-0 vs the datagen beam),
                # so the warm-started value head stays the oracle critic and doesn't collapse.
                value_tgt[t, i] = _max_cand_value(searcher, envs[i]) / vs
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

        # Value target = oracle evaluate_state (phi), collected per move above. No discounted-return
        # bootstrap: the value matches the pretraining objective so warm-start doesn't collapse.
        sel = storable.reshape(-1)
        if not sel.any():
            print(
                f"Gen {gen}: no storable samples (all dead); skipping update.",
                flush=True,
            )
            continue

        buf = {
            "boards": tf.constant(_flat(boards, sel), tf.float32),
            "pieces": tf.constant(_flat(pieces, sel), tf.int64),
            "bcg": tf.constant(_flat(bcg, sel), tf.float32),
            "cand_placements": tf.constant(_flat(cand_pl, sel), tf.float32),
            "cand_mask": tf.constant(_flat(cand_mk, sel), tf.bool),
            "pi_target": tf.constant(_flat(pi_tgt, sel), tf.float32),
            "value_target": tf.constant(_flat(value_tgt, sel), tf.float32),
        }
        ds = (
            tf.data.Dataset.from_tensor_slices(buf)
            .shuffle(int(sel.sum()))
            .batch(mini_batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        updates = 0
        step_out = None
        for _ in range(num_epochs):
            for batch in ds:
                step_out = train_step(net, batch, tf.constant(value_coef, tf.float32))
                updates += 1
        if step_out is None:
            print(f"Gen {gen}: buffer smaller than batch; skipping update.", flush=True)
            continue

        raw_returns = compute_raw_returns(
            rewards[..., None], dones[..., None], cfg.gamma, horizon, num_games
        )
        return_var = 0.99 * return_var + 0.01 * float(
            tf.math.reduce_variance(raw_returns)
        )
        return_scale.assign(tf.sqrt(return_var))

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
