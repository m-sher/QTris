from qtris.runners.flat import FlatRunner
from qtris.models.ar.model import ValueModel
from qtris.models.flat.model import FlatPolicyModel
from qtris.pretraining.flat import FlatPretrainer
from TetrisEnv.Moves import Keys
import tensorflow as tf
from tensorflow_probability import distributions
from tensorflow import keras
import tf_agents
import wandb
import time
import os

HARD_DROP_ID = Keys.HARD_DROP

# Model params
piece_dim = 8
key_dim = 12
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.0
max_len = 15
num_row_tiers = 2
num_sequences = 160 * num_row_tiers

# Environment params
generations = 1_000_000
num_envs = 64
num_collection_steps = 256
queue_size = 5
max_holes = 50
max_height = 18
max_steps = 1024
garbage_chance_min = 0.15
garbage_chance_max = 0.15
garbage_rows_min = 1
garbage_rows_max = 4

# Training params
mini_batch_size = 512
num_epochs = 4
num_updates = num_epochs * num_envs * num_collection_steps // mini_batch_size

gamma = 0.99
lam = 0.95
ppo_clip = 0.2
value_clip = 0.5
entropy_coef = 0.01
temperature = 1.0

target_kl = 0.02
early_stopping = True

expert_coef = 0.1
expert_dataset_path = "datasets/tetris_expert_dataset_flat"

config = {
    "num_envs": num_envs,
    "num_collection_steps": num_collection_steps,
    "mini_batch_size": mini_batch_size,
    "num_updates": num_updates,
    "gamma": gamma,
    "lam": lam,
    "ppo_clip": ppo_clip,
    "value_clip": value_clip,
    "entropy_coef": entropy_coef,
    "target_kl": target_kl,
    "early_stopping": early_stopping,
    "expert_coef": expert_coef,
}


@tf.function(jit_compile=True)
def compute_gae_and_returns(values, last_values, rewards, dones, gamma, lam):
    advantages = tf.TensorArray(
        dtype=tf.float32, size=num_collection_steps, element_shape=(num_envs, 1)
    )

    last_adv = tf.zeros(advantages.element_shape, dtype=tf.float32)
    last_val = last_values

    for t in tf.range(num_collection_steps - 1, -1, -1):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_val * mask - values[t]
        last_adv = delta + gamma * lam * last_adv * mask
        advantages = advantages.write(t, last_adv)
        last_val = values[t]

    advantages = tf.ensure_shape(
        advantages.stack(), (num_collection_steps, num_envs, 1)
    )

    returns = tf.ensure_shape(advantages + values, (num_collection_steps, num_envs, 1))

    return advantages, returns


@tf.function(jit_compile=True)
def compute_raw_returns(rewards, dones, gamma):
    returns = tf.TensorArray(
        dtype=tf.float32, size=num_collection_steps, element_shape=(num_envs, 1)
    )

    last_ret = tf.zeros(returns.element_shape, dtype=tf.float32)

    for t in tf.range(num_collection_steps - 1, -1, -1):
        mask = 1.0 - dones[t]
        last_ret = rewards[t] + gamma * last_ret * mask
        returns = returns.write(t, last_ret)

    return tf.ensure_shape(returns.stack(), (num_collection_steps, num_envs, 1))


@tf.function()
def train_step(
    p_model,
    v_model,
    online_batch,
    entropy_coef,
    expert_coef,
    use_expert,
    expert_batch=None,
):
    online_board_batch = tf.ensure_shape(
        online_batch["boards"], (mini_batch_size, 24, 10, 1)
    )
    online_pieces_batch = tf.ensure_shape(
        online_batch["pieces"], (mini_batch_size, queue_size + 2)
    )
    online_b2b_combo_garbage_batch = tf.ensure_shape(
        online_batch["b2b_combo_garbage"], (mini_batch_size, 3)
    )
    online_valid_sequences_batch = tf.ensure_shape(
        online_batch["valid_sequences"], (mini_batch_size, num_sequences, max_len)
    )
    action_indices_batch = tf.ensure_shape(
        online_batch["action_indices"], (mini_batch_size,)
    )

    old_log_probs_batch = tf.ensure_shape(
        online_batch["old_log_probs"], (mini_batch_size, 1)
    )
    advantages_batch = tf.ensure_shape(online_batch["advantages"], (mini_batch_size, 1))
    returns_batch = tf.ensure_shape(online_batch["returns"], (mini_batch_size, 1))
    old_values_batch = tf.ensure_shape(online_batch["old_values"], (mini_batch_size, 1))

    valid_mask = tf.reduce_any(
        tf.equal(
            online_valid_sequences_batch,
            tf.constant(HARD_DROP_ID, dtype=tf.int64),
        ),
        axis=-1,
    )

    with tf.GradientTape() as p_tape:
        logits, piece_scores = p_model(
            (
                online_board_batch,
                online_pieces_batch,
                online_b2b_combo_garbage_batch,
            ),
            training=True,
            return_scores=True,
        )

        masked_logits = tf.where(
            valid_mask, logits / temperature, tf.constant(-1e9, dtype=tf.float32)
        )

        dist = distributions.Categorical(logits=masked_logits, dtype=tf.int64)

        new_log_probs = tf.ensure_shape(
            dist.log_prob(action_indices_batch)[..., None], (mini_batch_size, 1)
        )

        ratio = tf.ensure_shape(
            tf.exp(new_log_probs - old_log_probs_batch), (mini_batch_size, 1)
        )
        clipped_ratio = tf.ensure_shape(
            tf.clip_by_value(ratio, 1 - ppo_clip, 1 + ppo_clip),
            (mini_batch_size, 1),
        )

        surr1 = tf.ensure_shape(ratio * advantages_batch, (mini_batch_size, 1))
        surr2 = tf.ensure_shape(
            clipped_ratio * advantages_batch,
            (mini_batch_size, 1),
        )

        ppo_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy = tf.reduce_mean(dist.entropy())

        approx_kl = tf.reduce_mean(old_log_probs_batch - new_log_probs)

        # Expert BC anchor
        if use_expert:
            expert_action_indices = tf.ensure_shape(
                expert_batch["action_indices"], (mini_batch_size,)
            )
            expert_valid_masks = tf.ensure_shape(
                expert_batch["valid_masks"], (mini_batch_size, num_sequences)
            )
            expert_sample_weights = tf.ensure_shape(
                expert_batch["sample_weights"], (mini_batch_size,)
            )

            expert_logits = p_model(
                (
                    tf.ensure_shape(expert_batch["boards"], (mini_batch_size, 24, 10, 1)),
                    tf.ensure_shape(expert_batch["pieces"], (mini_batch_size, queue_size + 2)),
                    tf.ensure_shape(expert_batch["b2b_combo_garbage"], (mini_batch_size, 3)),
                ),
                training=True,
            )
            expert_masked_logits = tf.where(
                expert_valid_masks, expert_logits, tf.constant(-1e9, dtype=tf.float32)
            )
            expert_per_sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=expert_action_indices, logits=expert_masked_logits
            )
            expert_loss = tf.math.divide_no_nan(
                tf.reduce_sum(expert_per_sample_loss * expert_sample_weights),
                tf.reduce_sum(expert_sample_weights),
            )

            expert_pred = tf.argmax(expert_masked_logits, axis=-1, output_type=tf.int64)
            expert_accuracy = tf.reduce_mean(
                tf.cast(expert_pred == expert_action_indices, tf.float32)
            )
        else:
            expert_loss = tf.constant(0.0, dtype=tf.float32)
            expert_accuracy = tf.constant(0.0, dtype=tf.float32)

        total_policy_loss = ppo_loss - entropy_coef * entropy + expert_coef * expert_loss

    p_gradients = p_tape.gradient(total_policy_loss, p_model.trainable_variables)
    p_model.optimizer.apply_gradients(zip(p_gradients, p_model.trainable_variables))

    clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32))

    with tf.GradientTape() as v_tape:
        values = v_model(
            (online_board_batch, online_pieces_batch, online_b2b_combo_garbage_batch),
            training=True,
        )

        value_error = tf.ensure_shape(values - returns_batch, (mini_batch_size, 1))
        clipped_values = old_values_batch + tf.clip_by_value(
            values - old_values_batch, -value_clip, value_clip
        )
        clipped_value_error = clipped_values - returns_batch
        value_loss = tf.reduce_mean(
            tf.maximum(tf.square(value_error), tf.square(clipped_value_error))
        )

    v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
    v_model.optimizer.apply_gradients(zip(v_gradients, v_model.trainable_variables))

    ret_var = tf.math.reduce_variance(returns_batch)
    res_var = tf.math.reduce_variance(returns_batch - values)
    explained_var = tf.reduce_mean(1.0 - tf.math.divide_no_nan(res_var, ret_var))

    return {
        "ppo_loss": ppo_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipped_frac": clipped_frac,
        "value_loss": value_loss,
        "explained_var": explained_var,
        "board": online_board_batch[0],
        "scores": piece_scores,
        "expert_loss": expert_loss,
        "expert_accuracy": expert_accuracy,
    }


def train_on_dataset(p_model, v_model, online_dataset, expert_iter, num_epochs, entropy_coef, expert_coef):
    use_expert = expert_iter is not None
    for epoch in range(num_epochs):
        for online_batch in online_dataset:
            if use_expert:
                expert_batch = next(expert_iter)
                step_out = train_step(
                    p_model, v_model, online_batch,
                    entropy_coef, expert_coef, True, expert_batch,
                )
            else:
                step_out = train_step(
                    p_model, v_model, online_batch,
                    entropy_coef, expert_coef, False,
                )

            if early_stopping and step_out["approx_kl"] >= 1.5 * target_kl:
                return step_out

    return step_out


def main(args):
    p_model = FlatPolicyModel(
        batch_size=num_envs,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        num_sequences=num_sequences,
    )

    v_model = ValueModel(
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=1,
    )

    print("Initialized models", flush=True)

    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )

    v_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    print("Built models", flush=True)

    p_optimizer = keras.optimizers.Adam(3e-5, clipnorm=0.5)
    p_model.compile(optimizer=p_optimizer, jit_compile=True)

    v_optimizer = keras.optimizers.Adam(3e-5, clipnorm=0.5)
    v_model.compile(optimizer=v_optimizer, jit_compile=True)

    p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint, "checkpoints/flat_policy", max_to_keep=3
    )

    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored from flat head policy checkpoint", flush=True)
    else:
        ar_ckpt = tf.train.Checkpoint(model=p_model)
        ar_mgr = tf.train.CheckpointManager(
            ar_ckpt, "checkpoints/ar_policy", max_to_keep=1
        )
        if ar_mgr.latest_checkpoint:
            ar_ckpt.restore(ar_mgr.latest_checkpoint).expect_partial()
            print("Restored encoder from autoregressive checkpoint", flush=True)
        else:
            print("No checkpoint found, training from scratch", flush=True)

    loaded_return_scale = tf.Variable(
        1.0, trainable=False, dtype=tf.float32, name="return_scale"
    )
    v_checkpoint = tf.train.Checkpoint(
        model=v_model,
        optimizer=v_optimizer,
        return_scale=loaded_return_scale,
    )
    v_checkpoint_manager = tf.train.CheckpointManager(
        v_checkpoint, "checkpoints/ar_value", max_to_keep=3
    )

    if v_checkpoint_manager.latest_checkpoint:
        v_checkpoint.restore(v_checkpoint_manager.latest_checkpoint).expect_partial()
        print(
            f"Restored value checkpoint (return_scale={float(loaded_return_scale):.3f})",
            flush=True,
        )
    else:
        pretrained_v_ckpt = tf.train.Checkpoint(
            model=v_model,
            return_scale=loaded_return_scale,
        )
        pretrained_v_mgr = tf.train.CheckpointManager(
            pretrained_v_ckpt, "checkpoints/flat_pretrained_value", max_to_keep=1
        )
        if pretrained_v_mgr.latest_checkpoint:
            pretrained_v_ckpt.restore(pretrained_v_mgr.latest_checkpoint).expect_partial()
            print(
                f"Bootstrapped value from pretrained checkpoint "
                f"(return_scale={float(loaded_return_scale):.3f})",
                flush=True,
            )
        else:
            print("No value checkpoint found, training from scratch", flush=True)

    p_model.summary()
    v_model.summary()

    runner = FlatRunner(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=max_steps,
        max_len=max_len,
        num_steps=num_collection_steps,
        num_envs=num_envs,
        garbage_chance_min=garbage_chance_min,
        garbage_chance_max=garbage_chance_max,
        garbage_rows_min=garbage_rows_min,
        garbage_rows_max=garbage_rows_max,
        p_model=p_model,
        v_model=v_model,
        temperature=temperature,
        seed=None,
        num_sequences=num_sequences,
        num_row_tiers=num_row_tiers,
    )

    print("Initialized runner", flush=True)
    last_time = time.time()

    wandb_run = wandb.init(
        project="Tetris",
        config=config,
    )

    if os.path.exists(expert_dataset_path):
        expert_dataset = FlatPretrainer.load_expert_dataset(expert_dataset_path, mini_batch_size)
        expert_iter = iter(expert_dataset)
        print(f"Loaded expert dataset from {expert_dataset_path}", flush=True)
    else:
        expert_iter = None
        print(
            f"No expert dataset found at {expert_dataset_path}; "
            f"running PPO without expert anchoring",
            flush=True,
        )

    return_var = float(loaded_return_scale) ** 2
    return_var_decay = 0.99
    print(f"Initial return_var = {return_var:.3f}", flush=True)

    for gen in range(args.num_generations):
        print(f"{time.time() - last_time:2.2f} | Collecting trajectory...", flush=True)
        last_time = time.time()

        (
            all_boards,
            all_pieces,
            all_b2b_combo_garbage,
            all_log_probs,
            all_valid_sequences,
            all_action_indices,
            all_values,
            all_last_values,
            all_attacks,
            all_clears,
            all_attack_reward,
            all_total_reward,
            all_dones,
            all_garbage_pushed,
        ) = runner.collect_trajectory(render=False)

        all_rewards = tf.ensure_shape(
            all_total_reward[..., None], (num_collection_steps, num_envs, 1)
        )

        scaled_rewards = tf.clip_by_value(
            all_rewards / (tf.sqrt(return_var) + 1e-8),
            -10.0, 10.0
        )

        print(
            f"{time.time() - last_time:2.2f} | Collected. Creating dataset...",
            flush=True,
        )
        last_time = time.time()

        all_advantages, all_returns = compute_gae_and_returns(
            all_values, all_last_values, scaled_rewards, all_dones, gamma, lam
        )

        raw_returns = compute_raw_returns(all_rewards, all_dones, gamma)
        batch_var = tf.math.reduce_variance(raw_returns)
        return_var = return_var_decay * return_var + (1 - return_var_decay) * batch_var

        all_advantages = (all_advantages - tf.reduce_mean(all_advantages)) / (
            tf.math.reduce_std(all_advantages) + 1e-8
        )

        boards_flat = tf.reshape(all_boards, (-1, 24, 10, 1))
        pieces_flat = tf.reshape(all_pieces, (-1, (queue_size + 2)))
        b2b_combo_garbage_flat = tf.reshape(all_b2b_combo_garbage, (-1, 3))
        valid_sequences_flat = tf.reshape(
            all_valid_sequences, (-1, num_sequences, max_len)
        )
        action_indices_flat = tf.reshape(all_action_indices, (-1,))
        log_probs_flat = tf.reshape(all_log_probs, (-1, 1))
        advantages_flat = tf.reshape(all_advantages, (-1, 1))
        returns_flat = tf.reshape(all_returns, (-1, 1))
        values_flat = tf.reshape(all_values, (-1, 1))

        online_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "boards": boards_flat,
                    "pieces": pieces_flat,
                    "b2b_combo_garbage": b2b_combo_garbage_flat,
                    "valid_sequences": valid_sequences_flat,
                    "action_indices": action_indices_flat,
                    "old_log_probs": log_probs_flat,
                    "advantages": advantages_flat,
                    "returns": returns_flat,
                    "old_values": values_flat,
                }
            )
            .cache()
            .shuffle(buffer_size=boards_flat.shape[0])
            .batch(
                mini_batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
                drop_remainder=True,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

        print(
            f"{time.time() - last_time:2.2f} | Dataset made. Training on dataset...",
            flush=True,
        )
        last_time = time.time()

        train_out = train_on_dataset(
            p_model, v_model, online_dataset, expert_iter, num_epochs, entropy_coef, expert_coef
        )

        if gen % 5 == 0:
            p_checkpoint_manager.save()
            v_checkpoint_manager.save()

        print(
            f"{time.time() - last_time:2.2f} | Trained on dataset. Logging metrics...",
            flush=True,
        )
        last_time = time.time()

        ppo_loss = train_out["ppo_loss"]
        entropy = train_out["entropy"]
        approx_kl = train_out["approx_kl"]
        clipped_frac = train_out["clipped_frac"]
        value_loss = train_out["value_loss"]
        explained_var = train_out["explained_var"]
        board = train_out["board"]
        scores = train_out["scores"]
        expert_loss = train_out["expert_loss"]
        expert_accuracy = train_out["expert_accuracy"]

        avg_reward = tf.reduce_mean(tf.reduce_sum(all_rewards, axis=0))
        avg_attacks = tf.reduce_mean(tf.reduce_sum(all_attacks, axis=0))
        avg_clears = tf.reduce_mean(tf.reduce_sum(all_clears, axis=0))
        avg_attack_reward = tf.reduce_mean(tf.reduce_sum(all_attack_reward, axis=0))
        avg_total_reward = tf.reduce_mean(tf.reduce_sum(all_total_reward, axis=0))
        avg_deaths = tf.reduce_mean(tf.reduce_sum(all_dones, axis=0))
        avg_pieces = tf.reduce_mean(
            num_collection_steps / (tf.reduce_sum(all_dones, axis=0) + 1)
        )
        avg_probs = tf.reduce_mean(tf.exp(all_log_probs))
        avg_garbage_pushed = tf.reduce_mean(tf.reduce_sum(all_garbage_pushed, axis=0))

        b2b_series = all_b2b_combo_garbage[..., 0]
        combo_series = all_b2b_combo_garbage[..., 1]
        avg_b2b = tf.reduce_mean(b2b_series)
        max_b2b = tf.reduce_max(b2b_series)
        avg_combo = tf.reduce_mean(combo_series)
        surge_rate = tf.reduce_mean(tf.cast(b2b_series >= 4, tf.float32))

        c_scores = tf.reshape(tf.reduce_mean(scores, axis=[0, 2, 3])[0, :60], (12, 5, 1))
        norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (
            tf.reduce_max(c_scores) - tf.reduce_min(c_scores)
        )

        if gen % 4 == 0:
            wandb.log(
                {
                    "ppo_loss": ppo_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clipped_frac": clipped_frac,
                    "value_loss": value_loss,
                    "explained_var": explained_var,
                    "return_var": return_var,
                    "avg_probs": avg_probs,
                    "avg_reward": avg_reward,
                    "avg_attacks": avg_attacks,
                    "avg_clears": avg_clears,
                    "avg_attack_reward": avg_attack_reward,
                    "avg_total_reward": avg_total_reward,
                    "avg_garbage_pushed": avg_garbage_pushed,
                    "avg_deaths": avg_deaths,
                    "avg_pieces": avg_pieces,
                    "avg_b2b": avg_b2b,
                    "max_b2b": max_b2b,
                    "avg_combo": avg_combo,
                    "surge_rate": surge_rate,
                    "expert_loss": expert_loss,
                    "expert_accuracy": expert_accuracy,
                    "expert_coef": expert_coef,
                    "board": wandb.Image(board[..., 0]),
                    "scores": wandb.Image(norm_c_scores),
                }
            )

        print(
            f"{time.time() - last_time:2.2f} | Gen: {gen} | Reward: {avg_reward}",
            flush=True,
        )
        last_time = time.time()

    runner.env.close()
    wandb_run.finish()
