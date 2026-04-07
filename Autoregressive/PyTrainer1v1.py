from TetrisEnv.Py1v1TetrisRunnerFlat import Py1v1TetrisRunnerFlat
from TetrisEnv.Moves import Keys
from TetrisModel import PolicyModel, AsymmetricValueModel
import tensorflow as tf
from tensorflow_probability import distributions
from tensorflow import keras
import tf_agents
import wandb
import time
import os
import glob
import random

HARD_DROP_ID = Keys.HARD_DROP

# Model params
piece_dim = 8
key_dim = 12
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.00
max_len = 15

# Environment params
generations = 1_000_000
num_envs = 64
num_collection_steps = 256
queue_size = 5
max_holes = 50
max_height = 18
max_steps = 512
num_row_tiers = 2
num_sequences = 160 * num_row_tiers

# Training params
mini_batch_size = 512
p_micro_batch_size = 64  # policy gradient accumulation chunk (64 * 320 seqs ≈ 4 GB)
num_epochs = 4
num_updates = num_epochs * num_envs * num_collection_steps // mini_batch_size
early_stopping = True

gamma = 0.99
lam = 0.95
ppo_clip = 0.2
value_clip = 0.5
entropy_coef = 1e-4
temperature = 1.0

target_kl = 0.02

# B2B gap shaping
b2b_gap_coef = 1.0

# Opponent pool params
pool_save_interval = 25
max_pool_size = 50
pool_dir = "./opponent_pool"

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
    "b2b_gap_coef": b2b_gap_coef,
    "pool_save_interval": pool_save_interval,
    "max_pool_size": max_pool_size,
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


_num_p_micros = mini_batch_size // p_micro_batch_size


@tf.function()
def train_step(p_model, v_model, online_batch, entropy_coef):
    boards = tf.ensure_shape(
        online_batch["boards"], (mini_batch_size, 24, 10, 1)
    )
    pieces = tf.ensure_shape(
        online_batch["pieces"], (mini_batch_size, queue_size + 2)
    )
    bcg = tf.ensure_shape(
        online_batch["b2b_combo_garbage"], (mini_batch_size, 3)
    )
    valid_seqs = tf.ensure_shape(
        online_batch["valid_sequences"], (mini_batch_size, num_sequences, max_len)
    )
    action_indices = tf.ensure_shape(
        online_batch["action_indices"], (mini_batch_size,)
    )
    old_log_probs = tf.ensure_shape(
        online_batch["old_log_probs"], (mini_batch_size, 1)
    )

    # Normalise advantages over the full mini-batch before splitting
    advantages = tf.ensure_shape(online_batch["advantages"], (mini_batch_size, 1))
    advantages = (
        (advantages - tf.reduce_mean(advantages))
        / (tf.math.reduce_std(advantages) + 1e-9)
    )

    returns = tf.ensure_shape(online_batch["returns"], (mini_batch_size, 1))
    old_values = tf.ensure_shape(online_batch["old_values"], (mini_batch_size, 1))

    # Opponent state for asymmetric value model
    opp_boards = tf.ensure_shape(
        online_batch["opp_boards"], (mini_batch_size, 24, 10, 1)
    )
    opp_pieces = tf.ensure_shape(
        online_batch["opp_pieces"], (mini_batch_size, queue_size + 2)
    )
    opp_bcg = tf.ensure_shape(
        online_batch["opp_b2b_combo_garbage"], (mini_batch_size, 3)
    )

    valid_mask = tf.reduce_any(
        tf.equal(valid_seqs, tf.constant(HARD_DROP_ID, dtype=tf.int64)),
        axis=-1,
    )

    # ------------------------------------------------------------------
    # Policy update — gradient accumulation over micro-batches
    # (keeps peak memory at p_micro_batch_size * num_sequences through decoder)
    # ------------------------------------------------------------------
    accum_p_grads = [tf.zeros_like(v) for v in p_model.trainable_variables]
    total_ppo_loss = tf.constant(0.0)
    total_entropy = tf.constant(0.0)
    total_approx_kl = tf.constant(0.0)
    total_clipped_frac = tf.constant(0.0)

    for m in range(_num_p_micros):
        s = m * p_micro_batch_size
        e = s + p_micro_batch_size

        with tf.GradientTape() as p_tape:
            piece_dec, piece_scores = p_model.process_obs(
                (boards[s:e], pieces[s:e], bcg[s:e]), training=True,
            )
            logits = p_model.score_sequences(
                piece_dec, valid_seqs[s:e], training=True,
            )

            masked_logits = tf.where(
                valid_mask[s:e],
                logits / temperature,
                tf.constant(-1e9, dtype=tf.float32),
            )
            dist = distributions.Categorical(
                logits=masked_logits, dtype=tf.int64,
            )
            new_log_probs = dist.log_prob(action_indices[s:e])[..., None]

            ratio = tf.exp(new_log_probs - old_log_probs[s:e])
            clipped_ratio = tf.clip_by_value(
                ratio, 1 - ppo_clip, 1 + ppo_clip,
            )

            surr1 = ratio * advantages[s:e]
            surr2 = clipped_ratio * advantages[s:e]
            ppo_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            entropy = tf.reduce_mean(dist.entropy())
            micro_loss = ppo_loss - entropy_coef * entropy

        grads = p_tape.gradient(micro_loss, p_model.trainable_variables)
        accum_p_grads = [a + g for a, g in zip(accum_p_grads, grads)]

        total_ppo_loss += ppo_loss
        total_entropy += entropy
        total_approx_kl += tf.reduce_mean(old_log_probs[s:e] - new_log_probs)
        total_clipped_frac += tf.reduce_mean(
            tf.cast(ratio != clipped_ratio, tf.float32)
        )

    # Average accumulated gradients and apply
    inv = 1.0 / _num_p_micros
    p_model.optimizer.apply_gradients(
        zip([g * inv for g in accum_p_grads], p_model.trainable_variables)
    )

    ppo_loss = total_ppo_loss * inv
    entropy = total_entropy * inv
    approx_kl = total_approx_kl * inv
    clipped_frac = total_clipped_frac * inv

    # ------------------------------------------------------------------
    # Value update — full batch (no sequence expansion, much cheaper)
    # ------------------------------------------------------------------
    with tf.GradientTape() as v_tape:
        values = v_model(
            (boards, pieces, bcg, opp_boards, opp_pieces, opp_bcg),
            training=True,
        )

        value_error = values - returns
        clipped_values = old_values + tf.clip_by_value(
            values - old_values, -value_clip, value_clip
        )
        clipped_value_error = clipped_values - returns
        value_loss = tf.reduce_mean(
            tf.maximum(tf.square(value_error), tf.square(clipped_value_error))
        )

    v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
    v_model.optimizer.apply_gradients(zip(v_gradients, v_model.trainable_variables))

    ret_var = tf.math.reduce_variance(returns)
    res_var = tf.math.reduce_variance(returns - values)
    explained_var = tf.reduce_mean(1.0 - tf.math.divide_no_nan(res_var, ret_var))

    return {
        "ppo_loss": ppo_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipped_frac": clipped_frac,
        "value_loss": value_loss,
        "explained_var": explained_var,
        "board": boards[0],
        "scores": piece_scores,
    }


def train_on_dataset(p_model, v_model, online_dataset, num_epochs, entropy_coef):
    for epoch in range(num_epochs):
        for online_batch in online_dataset:
            step_out = train_step(p_model, v_model, online_batch, entropy_coef)
            if early_stopping and step_out["approx_kl"] >= 1.5 * target_kl:
                return step_out

    return step_out


def save_pool_checkpoint(p_model, gen):
    """Save a snapshot of the current policy weights to the opponent pool."""
    os.makedirs(pool_dir, exist_ok=True)
    path = os.path.join(pool_dir, f"gen_{gen}")
    p_model.save_weights(path)

    # Prune oldest if over max_pool_size
    existing = sorted(glob.glob(os.path.join(pool_dir, "gen_*.index")))
    while len(existing) > max_pool_size:
        prefix = existing.pop(0).replace(".index", "")
        for f in glob.glob(prefix + ".*"):
            os.remove(f)


def load_pool_opponent(opp_model):
    """Sample an opponent from the pool, weighted toward recent checkpoints."""
    existing = sorted(glob.glob(os.path.join(pool_dir, "gen_*.index")))
    if not existing:
        return False

    # Linear weighting: newest gets highest weight
    n = len(existing)
    weights = list(range(1, n + 1))
    chosen = random.choices(existing, weights=weights, k=1)[0]
    prefix = chosen.replace(".index", "")

    opp_model.load_weights(prefix)
    gen_num = os.path.basename(prefix)
    print(f"Loaded opponent from {gen_num}", flush=True)
    return True


def main(argv):
    # -----------------------------------------------------------------------
    # Initialize models
    # -----------------------------------------------------------------------
    def _make_policy():
        return PolicyModel(
            batch_size=num_envs,
            piece_dim=piece_dim,
            key_dim=key_dim,
            depth=depth,
            max_len=max_len,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            output_dim=key_dim,
        )

    p_model = _make_policy()
    opp_model = _make_policy()

    v_model = AsymmetricValueModel(
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=1,
    )

    print("Initialized models", flush=True)

    p_optimizer = keras.optimizers.Adam(3e-5, clipnorm=0.5)
    p_model.compile(optimizer=p_optimizer, jit_compile=True)

    v_optimizer = keras.optimizers.Adam(3e-5, clipnorm=0.5)
    v_model.compile(optimizer=v_optimizer, jit_compile=True)

    # -----------------------------------------------------------------------
    # Checkpoint paths
    # -----------------------------------------------------------------------
    p_ckpt_dir = "./1v1_policy_checkpoints"
    solo_bootstrap_dir = "./policy_checkpoints"
    v_ckpt_dir = "./1v1_value_checkpoints"

    # Initialize checkpoint managers for training models
    p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint, p_ckpt_dir, max_to_keep=3
    )

    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"Restored policy from 1v1 checkpoint ({p_ckpt_dir})", flush=True)
    else:
        # Bootstrap from solo trainer's policy checkpoint
        solo_p_checkpoint = tf.train.Checkpoint(model=p_model)
        solo_p_manager = tf.train.CheckpointManager(
            solo_p_checkpoint, solo_bootstrap_dir, max_to_keep=1
        )
        if solo_p_manager.latest_checkpoint:
            solo_p_checkpoint.restore(solo_p_manager.latest_checkpoint).expect_partial()
            print(f"Bootstrapped policy from solo checkpoint ({solo_bootstrap_dir})", flush=True)
        else:
            print("No policy checkpoints found, starting from scratch", flush=True)

    v_checkpoint = tf.train.Checkpoint(model=v_model, optimizer=v_optimizer)
    v_checkpoint_manager = tf.train.CheckpointManager(
        v_checkpoint, v_ckpt_dir, max_to_keep=3
    )
    v_checkpoint.restore(v_checkpoint_manager.latest_checkpoint).expect_partial()
    print("Restored checkpoints", flush=True)

    # -----------------------------------------------------------------------
    # Build models (4-input call initialises all layers incl. key decoder)
    # -----------------------------------------------------------------------
    p_build_inputs = (
        keras.Input(shape=(24, 10, 1), dtype=tf.float32),
        keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
        keras.Input(shape=(3,), dtype=tf.float32),
        keras.Input(shape=(max_len,), dtype=tf.int64),
    )

    p_model(p_build_inputs)
    opp_model(p_build_inputs)

    v_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    print("Built models", flush=True)

    p_model.summary()
    v_model.summary()

    # Initialize opponent: copy current policy weights (or load from pool)
    if not load_pool_opponent(opp_model):
        opp_model.set_weights(p_model.get_weights())
        print("No pool checkpoints found, opponent initialized from current policy", flush=True)

    # -----------------------------------------------------------------------
    # Initialize runner
    # -----------------------------------------------------------------------
    runner = Py1v1TetrisRunnerFlat(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=max_steps,
        max_len=max_len,
        num_steps=num_collection_steps,
        num_envs=num_envs,
        p_model=p_model,
        opp_model=opp_model,
        v_model=v_model,
        temperature=temperature,
        seed=None,
        num_sequences=num_sequences,
        num_row_tiers=num_row_tiers,
        b2b_gap_coef=b2b_gap_coef,
    )

    print("Initialized runner", flush=True)
    last_time = time.time()

    # Initialize WandB logging
    wandb_run = wandb.init(
        project="Tetris-1v1",
        config=config,
    )

    # Initialize running return variance for reward scaling (EMA)
    return_var = 100.0
    return_var_decay = 0.99

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for gen in range(generations):
        # Collect trajectory
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
            all_wins,
            all_opp_boards,
            all_opp_pieces,
            all_opp_b2b_combo_garbage,
        ) = runner.collect_trajectory(render=False)

        all_rewards = tf.ensure_shape(
            all_total_reward[..., None], (num_collection_steps, num_envs, 1)
        )

        # Scale rewards by running return std
        scaled_rewards = tf.clip_by_value(
            all_rewards / (tf.sqrt(return_var) + 1e-9),
            -25.0, 25.0
        )

        print(
            f"{time.time() - last_time:2.2f} | Collected. Creating dataset...",
            flush=True,
        )
        last_time = time.time()

        # Compute advantages and returns
        all_advantages, all_returns = compute_gae_and_returns(
            all_values, all_last_values, scaled_rewards, all_dones, gamma, lam
        )

        # Update running return variance (EMA)
        batch_var = tf.math.reduce_variance(all_returns)
        return_var = return_var_decay * return_var + (1 - return_var_decay) * batch_var

        # Flatten data
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
        opp_boards_flat = tf.reshape(all_opp_boards, (-1, 24, 10, 1))
        opp_pieces_flat = tf.reshape(all_opp_pieces, (-1, (queue_size + 2)))
        opp_b2b_combo_garbage_flat = tf.reshape(all_opp_b2b_combo_garbage, (-1, 3))

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
                    "opp_boards": opp_boards_flat,
                    "opp_pieces": opp_pieces_flat,
                    "opp_b2b_combo_garbage": opp_b2b_combo_garbage_flat,
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

        # Train on collected data
        train_out = train_on_dataset(
            p_model, v_model, online_dataset, num_epochs, entropy_coef
        )

        # Save training checkpoints
        p_checkpoint_manager.save()
        v_checkpoint_manager.save()

        print(
            f"{time.time() - last_time:2.2f} | Trained on dataset. Logging metrics...",
            flush=True,
        )
        last_time = time.time()

        # Unpack metrics
        ppo_loss = train_out["ppo_loss"]
        entropy = train_out["entropy"]
        approx_kl = train_out["approx_kl"]
        clipped_frac = train_out["clipped_frac"]
        value_loss = train_out["value_loss"]
        explained_var = train_out["explained_var"]
        board = train_out["board"]
        scores = train_out["scores"]

        # Compute more metrics
        avg_reward = tf.reduce_mean(tf.reduce_sum(all_rewards, axis=0))
        avg_attacks = tf.reduce_mean(tf.reduce_sum(all_attacks, axis=0))
        avg_clears = tf.reduce_mean(tf.reduce_sum(all_clears, axis=0))
        avg_attack_reward = tf.reduce_mean(tf.reduce_sum(all_attack_reward, axis=0))
        avg_total_reward = tf.reduce_mean(tf.reduce_sum(all_total_reward, axis=0))
        avg_garbage_pushed = tf.reduce_mean(tf.reduce_sum(all_garbage_pushed, axis=0))
        avg_deaths = tf.reduce_mean(tf.reduce_sum(all_dones, axis=0))
        avg_pieces = tf.reduce_mean(
            num_collection_steps / (tf.reduce_sum(all_dones, axis=0) + 1)
        )

        avg_probs = tf.reduce_mean(tf.exp(all_log_probs))

        c_scores = tf.reshape(tf.reduce_mean(scores, axis=[0, 2, 3])[0], (12, 5, 1))
        norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (
            tf.reduce_max(c_scores) - tf.reduce_min(c_scores)
        )

        # 1v1 metrics
        total_wins = tf.reduce_sum(all_wins)
        total_episodes = tf.reduce_sum(all_dones)
        total_losses = total_episodes - total_wins
        win_rate = tf.math.divide_no_nan(total_wins, total_episodes)

        # Save to opponent pool when win rate exceeds threshold, then load a new challenger
        if gen % pool_save_interval == 0 and total_episodes > 0 and win_rate >= 0.55:
            save_pool_checkpoint(p_model, gen)
            print(f"Saved to opponent pool at gen {gen} (win rate: {win_rate:.0%})", flush=True)
            if not load_pool_opponent(opp_model):
                opp_model.set_weights(p_model.get_weights())

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
                "win_rate": win_rate,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "board": wandb.Image(board[..., 0]),
                "scores": wandb.Image(norm_c_scores),
            }
        )

        print(
            f"{time.time() - last_time:2.2f} | Gen: {gen} | Reward: {avg_reward} | WR: {win_rate:.2f}",
            flush=True,
        )
        last_time = time.time()

    runner.env.close()
    wandb_run.finish()


if __name__ == "__main__":
    tf_agents.system.multiprocessing.handle_main(main)
