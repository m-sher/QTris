"""Single-player PPO for the placement model.

The action is a candidate placement slot; the value comes from the merged
`PlacementPolicyValueNet` (one net, optimizer, and tape). Warm-starts from the BC
checkpoint and fine-tunes on the model's own rollouts to close the open-loop
distribution-shift gap (see notes.md). wandb observability via `qtris.observability`
(SingleAgentTrainConfig + SingleAgentPPOLog).
"""

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import distributions

from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.observability.models import SingleAgentPPOLog, SingleAgentTrainConfig
from qtris.observability.wandb_backend import finish, init_run, log_step
from qtris.runners.placement import PlacementRunner
from qtris.training.gae import compute_gae_and_returns, compute_raw_returns
from qtris.training.ppo_loss import clipped_surrogate, clipped_value_loss

GAMMA = 0.99
LAM = 0.95
PPO_CLIP = 0.2
VALUE_CLIP = 0.5
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
TARGET_KL = 0.02
TEMPERATURE = 1.0

# Expert BC anchor: soft-CE to the oracle's cand_scores softmax, policy head only.
EXPERT_COEF = 1.0
EXPERT_TEMP = 1.0  # softmax temperature for the expert anchor target


def _load_expert_iter(path, batch_size):
    """Cycling iterator over the placement BC dataset for the PPO expert anchor."""
    if not os.path.exists(path):
        print(
            f"No expert dataset at {path}; running PPO without expert anchoring.",
            flush=True,
        )
        return None
    ds = tf.data.Dataset.load(path)
    if "cand_placements" not in ds.element_spec or "cand_scores" not in ds.element_spec:
        print(
            f"Dataset at {path} is not the placement schema; skipping expert anchor.",
            flush=True,
        )
        return None
    ds = (
        ds.cache()
        .repeat()
        .shuffle(100_000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    print(
        f"Loaded expert dataset {path} (soft-CE anchor, coef={EXPERT_COEF}).",
        flush=True,
    )
    return iter(ds)


def _sum_grads(rest_grads, expert_grads):
    """Per-variable sum of two gradient lists, densifying IndexedSlices (the embedding
    gradient is sparse, and IndexedSlices + IndexedSlices is unsupported)."""
    out = []
    for rg, eg in zip(rest_grads, expert_grads):
        if rg is None:
            out.append(eg)
        elif eg is None:
            out.append(rg)
        else:
            if isinstance(rg, tf.IndexedSlices):
                rg = tf.convert_to_tensor(rg)
            if isinstance(eg, tf.IndexedSlices):
                eg = tf.convert_to_tensor(eg)
            out.append(rg + eg)
    return out


@tf.function
def _rollout_grads(net, batch):
    """PPO + value + entropy on the rollout batch: one forward, one backward.
    Returns (gradients, metrics)."""
    cand_mask = batch["cand_mask"]
    with tf.GradientTape() as tape:
        logits, values, piece_scores = net(
            (
                batch["boards"],
                batch["pieces"],
                batch["bcg"],
                batch["cand_placements"],
                cand_mask,
            ),
            training=True,
            return_scores=True,
        )
        masked = tf.where(
            cand_mask, logits / TEMPERATURE, tf.constant(-1e9, tf.float32)
        )
        dist = distributions.Categorical(logits=masked, dtype=tf.int64)
        new_log_prob = dist.log_prob(batch["action_index"])
        ratio = tf.exp(new_log_prob - batch["old_log_prob"])
        surrogate, clipped_ratio = clipped_surrogate(
            ratio, batch["advantages"], PPO_CLIP
        )
        ppo_loss = -tf.reduce_mean(surrogate)
        entropy = tf.reduce_mean(dist.entropy())
        value_loss = clipped_value_loss(
            values[:, 0], batch["old_values"], batch["returns"], VALUE_CLIP
        )
        loss = ppo_loss - ENTROPY_COEF * entropy + VALUE_COEF * value_loss
    grads = tape.gradient(loss, net.trainable_variables)

    approx_kl = tf.reduce_mean(batch["old_log_prob"] - new_log_prob)
    clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32))
    ret_var = tf.math.reduce_variance(batch["returns"])
    res_var = tf.math.reduce_variance(batch["returns"] - values[:, 0])
    explained_var = 1.0 - tf.math.divide_no_nan(res_var, ret_var)
    metrics = {
        "ppo_loss": ppo_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipped_frac": clipped_frac,
        "value_loss": value_loss,
        "explained_var": explained_var,
        "board": batch["boards"][0],
        "scores": piece_scores,
    }
    return grads, metrics


@tf.function
def _expert_grads(net, expert_batch):
    """Expert BC anchor (soft-CE), policy head only: one forward, one backward.
    Returns (gradients, expert_loss, expert_accuracy)."""
    e_mask = expert_batch["cand_scores"] > -1e29
    e_masked = tf.where(
        e_mask, expert_batch["cand_scores"], tf.constant(-1e30, tf.float32)
    )
    e_target = tf.nn.softmax(e_masked / EXPERT_TEMP, axis=-1)
    with tf.GradientTape() as tape:
        e_logits, _ = net(
            (
                expert_batch["boards"],
                expert_batch["pieces"],
                expert_batch["b2b_combo_garbage"],
                expert_batch["cand_placements"],
                e_mask,
            ),
            training=True,
        )
        e_logp = tf.nn.log_softmax(e_logits, axis=-1)
        expert_loss = tf.reduce_mean(-tf.reduce_sum(e_target * e_logp, axis=-1))
        loss = EXPERT_COEF * expert_loss
    grads = tape.gradient(loss, net.trainable_variables)

    expert_pred = tf.argmax(e_logits, axis=-1, output_type=tf.int64)
    expert_label = tf.argmax(e_masked, axis=-1, output_type=tf.int64)
    expert_accuracy = tf.reduce_mean(tf.cast(expert_pred == expert_label, tf.float32))
    return grads, expert_loss, expert_accuracy


def train_step(net, batch, use_expert, expert_batch=None):
    """Rollout and expert gradients are computed in SEPARATE tf.functions (each one
    forward + backward, the known-good pattern) and summed before a single optimizer
    step. Two net() forwards inside ONE tf.function compiled to a NaN backward; keeping
    them in separate compiled graphs avoids it (see notes.md)."""
    grads, metrics = _rollout_grads(net, batch)
    if use_expert:
        e_grads, expert_loss, expert_accuracy = _expert_grads(net, expert_batch)
        grads = _sum_grads(grads, e_grads)
    else:
        expert_loss = tf.constant(0.0, tf.float32)
        expert_accuracy = tf.constant(0.0, tf.float32)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))
    metrics["expert_loss"] = expert_loss
    metrics["expert_accuracy"] = expert_accuracy
    return metrics


def main(args):
    piece_dim, depth, num_heads, num_layers = 8, 64, 4, 4
    queue_size, max_len = 5, 15
    num_envs, num_steps, mini_batch_size, num_epochs = 64, 64, 256, 4
    num_generations = getattr(args, "num_generations", 1_000_000)

    config = SingleAgentTrainConfig(
        num_envs=num_envs,
        num_collection_steps=num_steps,
        mini_batch_size=mini_batch_size,
        num_updates=num_epochs * num_envs * num_steps // mini_batch_size,
        gamma=GAMMA,
        lam=LAM,
        ppo_clip=PPO_CLIP,
        value_clip=VALUE_CLIP,
        entropy_coef=ENTROPY_COEF,
        target_kl=TARGET_KL,
        expert_coef=EXPERT_COEF,
    )

    net = PlacementPolicyValueNet(
        batch_size=num_envs,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=0.0,
    )
    optimizer = keras.optimizers.Adam(3e-5, clipnorm=0.5)
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
        model=net, optimizer=optimizer, return_scale=return_scale
    )
    manager = tf.train.CheckpointManager(
        checkpoint, "checkpoints/placement_rl", max_to_keep=3
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Resumed RL checkpoint {manager.latest_checkpoint}.", flush=True)
    else:
        warm = tf.train.latest_checkpoint("checkpoints/placement_pretrained_policy")
        if warm is not None:
            tf.train.Checkpoint(model=net).restore(warm).expect_partial()
            print(f"Warm-started from BC checkpoint {warm}.", flush=True)

    runner = PlacementRunner(
        queue_size=queue_size,
        max_holes=50,
        max_height=18,
        max_steps=1024,
        max_len=max_len,
        num_steps=num_steps,
        num_envs=num_envs,
        garbage_chance_min=0.0,
        garbage_chance_max=0.2,
        garbage_rows_min=1,
        garbage_rows_max=4,
        net=net,
        temperature=TEMPERATURE,
    )

    # No --expert-dataset -> no expert anchor (plain PPO).
    expert_path = getattr(args, "expert_dataset", None)
    expert_iter = (
        _load_expert_iter(expert_path, mini_batch_size) if expert_path else None
    )
    use_expert = expert_iter is not None

    wandb_run = init_run(project="Tetris", config=config)

    return_var = float(return_scale) ** 2
    B = num_steps * num_envs
    for gen in range(num_generations):
        buf = runner.collect_trajectory()
        rewards = buf["total_reward"][..., None]  # (T, N, 1)
        scaled = tf.clip_by_value(rewards / (tf.sqrt(return_var) + 1e-8), -10.0, 10.0)
        advantages, returns = compute_gae_and_returns(
            buf["values"],
            buf["last_values"],
            scaled,
            buf["dones"],
            GAMMA,
            LAM,
            num_steps,
            num_envs,
        )
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + 1e-8
        )

        ds = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "boards": tf.reshape(buf["boards"], (B, 24, 10, 1)),
                    "pieces": tf.reshape(buf["pieces"], (B, queue_size + 2)),
                    "bcg": tf.reshape(buf["bcg"], (B, 3)),
                    "cand_placements": tf.reshape(
                        buf["cand_placements"],
                        (B, CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM),
                    ),
                    "cand_mask": tf.reshape(buf["cand_mask"], (B, CANDIDATE_CAPACITY)),
                    "action_index": tf.reshape(buf["action_index"], (B,)),
                    "old_log_prob": tf.reshape(buf["log_prob"], (B,)),
                    "advantages": tf.reshape(advantages, (B,)),
                    "returns": tf.reshape(returns, (B,)),
                    "old_values": tf.reshape(buf["values"], (B,)),
                }
            )
            .shuffle(B)
            .batch(mini_batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        updates = 0
        step_out = None
        stop = False
        for _ in range(num_epochs):
            for batch in ds:
                expert_batch = next(expert_iter) if use_expert else None
                step_out = train_step(net, batch, use_expert, expert_batch)
                updates += 1
                if float(step_out["approx_kl"]) >= 1.5 * TARGET_KL:
                    stop = True
                    break
            if stop:
                break

        raw_returns = compute_raw_returns(
            rewards, buf["dones"], GAMMA, num_steps, num_envs
        )
        return_var = 0.99 * return_var + 0.01 * float(
            tf.math.reduce_variance(raw_returns)
        )
        return_scale.assign(tf.sqrt(return_var))

        # Aggregate gameplay/reward metrics from the trajectory buffer.
        avg_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=0))
        avg_attacks = tf.reduce_mean(tf.reduce_sum(buf["attacks"], axis=0))
        avg_clears = tf.reduce_mean(tf.reduce_sum(buf["clears"], axis=0))
        avg_attack_reward = tf.reduce_mean(tf.reduce_sum(buf["attack_reward"], axis=0))
        avg_total_reward = tf.reduce_mean(tf.reduce_sum(buf["total_reward"], axis=0))
        avg_garbage_pushed = tf.reduce_mean(
            tf.reduce_sum(buf["garbage_pushed"], axis=0)
        )
        avg_deaths = tf.reduce_mean(tf.reduce_sum(buf["dones"], axis=0))
        avg_pieces = tf.reduce_mean(
            num_steps / (tf.reduce_sum(buf["dones"], axis=0) + 1)
        )
        avg_probs = tf.reduce_mean(tf.exp(buf["log_prob"]))

        b2b_series = buf["bcg"][..., 0]
        combo_series = buf["bcg"][..., 1]
        avg_b2b = tf.reduce_mean(b2b_series)
        max_b2b = tf.reduce_max(b2b_series)
        avg_combo = tf.reduce_mean(combo_series)
        surge_rate = tf.reduce_mean(tf.cast(b2b_series >= 4, tf.float32))

        c_scores = tf.reshape(
            tf.reduce_mean(step_out["scores"], axis=[0, 2, 3])[0, :60], (12, 5, 1)
        )
        norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (
            tf.reduce_max(c_scores) - tf.reduce_min(c_scores)
        )

        if gen % 4 == 0:
            log_step(
                SingleAgentPPOLog(
                    ppo_loss=step_out["ppo_loss"],
                    entropy=step_out["entropy"],
                    approx_kl=step_out["approx_kl"],
                    clipped_frac=step_out["clipped_frac"],
                    value_loss=step_out["value_loss"],
                    explained_var=step_out["explained_var"],
                    return_var=return_var,
                    avg_probs=avg_probs,
                    avg_reward=avg_reward,
                    avg_attacks=avg_attacks,
                    avg_clears=avg_clears,
                    avg_attack_reward=avg_attack_reward,
                    avg_total_reward=avg_total_reward,
                    avg_garbage_pushed=avg_garbage_pushed,
                    avg_deaths=avg_deaths,
                    avg_pieces=avg_pieces,
                    avg_b2b=avg_b2b,
                    max_b2b=max_b2b,
                    avg_combo=avg_combo,
                    surge_rate=surge_rate,
                    expert_loss=step_out["expert_loss"],
                    expert_accuracy=step_out["expert_accuracy"],
                    expert_coef=EXPERT_COEF,
                    updates=updates,
                    board=step_out["board"][..., 0].numpy(),
                    scores=norm_c_scores.numpy(),
                )
            )
            print(
                f"Gen {gen} | Policy: {float(step_out['ppo_loss']):2.3f} | "
                f"Value: {float(step_out['value_loss']):2.3f} | "
                f"Ent: {float(step_out['entropy']):1.3f} | "
                f"KL: {float(step_out['approx_kl']):1.4f} | "
                f"Expert: {float(step_out['expert_loss']):2.3f} | "
                f"Reward: {float(avg_reward):3.1f} | Updates: {updates}",
                flush=True,
            )
        if gen % 5 == 0:
            manager.save()

    runner.env.close()
    finish(wandb_run)
