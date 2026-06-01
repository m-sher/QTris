"""Single-player PPO for the placement model (mirrors training/flat.py).

The action is a candidate placement slot; the value comes from the same merged
`PlacementPolicyValueNet`. Warm-starts from the BC checkpoint and fine-tunes on the
model's own rollouts to close the open-loop distribution-shift gap (see notes.md).
"""

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import distributions

from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
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

# Expert BC anchor (soft-CE to the oracle's cand_scores softmax; policy head only),
# mirroring the AR trainer's anchor and the placement BC pretraining objective.
EXPERT_COEF = 0.1
EXPERT_TEMP = 10.0  # matches placement pretraining policy_temp
EXPERT_DATASET_PATH = "datasets/tetris_oracle_placement"


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


@tf.function
def train_step(net, batch, use_expert, expert_batch=None):
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
        masked = tf.where(
            cand_mask, logits / TEMPERATURE, tf.constant(-1e9, tf.float32)
        )
        dist = distributions.Categorical(logits=masked, dtype=tf.int64)
        new_log_prob = dist.log_prob(batch["action_index"])
        ratio = tf.exp(new_log_prob - batch["old_log_prob"])
        surrogate, _ = clipped_surrogate(ratio, batch["advantages"], PPO_CLIP)
        ppo_loss = -tf.reduce_mean(surrogate)
        entropy = tf.reduce_mean(dist.entropy())
        value_loss = clipped_value_loss(
            values[:, 0], batch["old_values"], batch["returns"], VALUE_CLIP
        )

        # Expert BC anchor: soft-CE to the oracle's cand_scores softmax (the placement
        # pretraining target), policy head only. Keeps PPO from drifting off the manifold.
        if use_expert:
            e_mask = expert_batch["cand_scores"] > -1e29
            e_masked = tf.where(
                e_mask, expert_batch["cand_scores"], tf.constant(-1e30, tf.float32)
            )
            e_target = tf.nn.softmax(e_masked / EXPERT_TEMP, axis=-1)
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
        else:
            expert_loss = tf.constant(0.0, tf.float32)

        loss = (
            ppo_loss
            - ENTROPY_COEF * entropy
            + VALUE_COEF * value_loss
            + EXPERT_COEF * expert_loss
        )

    grads = tape.gradient(loss, net.trainable_variables)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))
    approx_kl = tf.reduce_mean(batch["old_log_prob"] - new_log_prob)
    return ppo_loss, value_loss, entropy, approx_kl, expert_loss


def main(args):
    piece_dim, depth, num_heads, num_layers = 8, 64, 4, 4
    queue_size, max_len = 5, 15
    num_envs, num_steps, mini_batch_size, num_epochs = 64, 64, 256, 4
    num_generations = getattr(args, "num_generations", 1_000_000)

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

    # argparse sets expert_dataset=None when the flag is omitted, so `or` resolves it to
    # the default (getattr's fallback wouldn't fire on an attribute that exists as None).
    expert_path = getattr(args, "expert_dataset", None) or EXPERT_DATASET_PATH
    expert_iter = _load_expert_iter(expert_path, mini_batch_size)
    use_expert = expert_iter is not None

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

        last = (0.0, 0.0, 0.0, 0.0, 0.0)
        stop = False
        for _ in range(num_epochs):
            for batch in ds:
                expert_batch = next(expert_iter) if use_expert else None
                pl, vl, ent, kl, el = train_step(net, batch, use_expert, expert_batch)
                last = (float(pl), float(vl), float(ent), float(kl), float(el))
                if last[3] >= 1.5 * TARGET_KL:
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

        if gen % 4 == 0:
            ep_reward = float(
                tf.reduce_sum(rewards) / (tf.reduce_sum(buf["dones"]) + 1.0)
            )
            print(
                f"Gen {gen} | Policy: {last[0]:2.3f} | Value: {last[1]:2.3f} | "
                f"Ent: {last[2]:1.3f} | KL: {last[3]:1.4f} | Expert: {last[4]:2.3f} | "
                f"ret/death: {ep_reward:3.1f}",
                flush=True,
            )
        if gen % 5 == 0:
            manager.save()
