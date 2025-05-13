from TetrisModel import PolicyModel, ValueModel
from TetrisEnvs.TFTetrisEnv.TFTetrisRunner import TFTetrisRunner
import tensorflow as tf
import multiprocessing as mp
import tf_agents
import keras
import pygame
import wandb
import time

# Model params
piece_dim = 8
key_dim = 12
depth = 32
num_heads = 4
num_layers = 4
dropout_rate = 0.1

# Environment params
generations = 10000000
num_envs = 1
num_collection_steps = 500
queue_size = 5
max_holes = 4

# Training params
mini_batch_size = 1024
epochs_per_gen = 10

gamma = 0.99
lam = 0.95
ppo_clip = 0.2
value_clip = 0.2

entropy_coef = 0.1

target_kl = 0.02
kl_tolerance = 0.3

max_len = 9

def main(argv):
    # Initialize models and optimizers
    p_model = PolicyModel(batch_size=num_envs,
                        piece_dim=piece_dim,
                        key_dim=key_dim,
                        depth=depth,
                        max_len=max_len,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate,
                        output_dim=key_dim)
    p_optimizer = keras.optimizers.Adam(1e-4)
    p_model.compile(optimizer=p_optimizer, jit_compile=True)

    v_model = ValueModel(piece_dim=piece_dim,
                        depth=depth,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate,
                        output_dim=1)

    v_optimizer = keras.optimizers.Adam(1e-4)
    v_model.compile(optimizer=v_optimizer, jit_compile=True)
    print("Initialized models and optimizers", flush=True)

    # Initialize checkpoint manager
    p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
    p_checkpoint_manager = tf.train.CheckpointManager(p_checkpoint, './combined_pretrain_checkpoints', max_to_keep=3)
    p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
    p_checkpoint_manager = tf.train.CheckpointManager(p_checkpoint, './policy_checkpoints', max_to_keep=3)

    v_checkpoint = tf.train.Checkpoint(model=v_model, optimizer=v_optimizer)
    # v_checkpoint_manager = tf.train.CheckpointManager(v_checkpoint, './combined_pretrain_checkpoints', max_to_keep=3)
    # v_checkpoint.restore(v_checkpoint_manager.latest_checkpoint).expect_partial()
    v_checkpoint_manager = tf.train.CheckpointManager(v_checkpoint, './value_checkpoints', max_to_keep=3)
    print("Restored checkpoint", flush=True)

    dummy_board = tf.random.uniform((num_envs, 24, 10, 1), maxval=2, dtype=tf.float32)
    dummy_pieces = tf.random.uniform((num_envs, queue_size + 2), maxval=piece_dim, dtype=tf.int64)

    p_model.build(input_shape=[(None, 24, 10, 1),
                            (None, queue_size + 2),
                            (None, max_len)])

    v_model.build(input_shape=[(None, 24, 10, 1),
                            (None, queue_size + 2)])

    p_model.summary()
    v_model.summary()

    # Initialize runner
    runner = TFTetrisRunner(queue_size=queue_size,
                            max_holes=max_holes,
                            max_len=max_len,
                            num_steps=num_collection_steps,
                            num_envs=num_envs,
                            key_dim=key_dim,
                            p_model=p_model,
                            v_model=v_model,
                            seed=[4, 0])

    start = time.time()
    (all_boards, all_pieces, all_actions, all_log_probs,
    all_values, all_attacks, all_height_penalty,
    all_hole_penalty, all_skyline_penalty,
    all_bumpy_penalty, all_death_penalty, all_dones) = runner.collect_trajectory()
    print(f"First run time: {time.time() - start}", flush=True)

    start = time.time()
    (all_boards, all_pieces, all_actions, all_log_probs,
    all_values, all_attacks, all_height_penalty,
    all_hole_penalty, all_skyline_penalty,
    all_bumpy_penalty, all_death_penalty, all_dones) = runner.collect_trajectory()
    print(f"Second run time: {time.time() - start}", flush=True)

    start = time.time()
    (all_boards, all_pieces, all_actions, all_log_probs,
    all_values, all_attacks, all_height_penalty,
    all_hole_penalty, all_skyline_penalty,
    all_bumpy_penalty, all_death_penalty, all_dones) = runner.collect_trajectory()
    print(f"Third run time: {time.time() - start}", flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    tf_agents.system.multiprocessing.handle_main(main)

# @tf.function(jit_compile=True)
# def compute_gae_and_returns(values, rewards, dones, gamma, lam):
#     """
#     Args:
#         values: Value predictions (num_collection_steps, num_envs)
#         rewards: Raw rewards received (num_collection_steps, num_envs)
#         dones: Episode terminations (num_collection_steps, num_envs)
#         gamma: Discount factor
#         lam: GAE lambda parameter

#     Returns:
#         advantages: GAE advantages (num_collection_steps, num_envs)
#         returns: Returns (num_collection_steps, num_envs)
#     """

#     advantages = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
#                                 element_shape=(num_envs,))

#     last_adv = tf.zeros(advantages.element_shape, dtype=tf.float32)
#     last_val = values[-1]

#     for t in tf.range(advantages.size() - 1, -1, -1):
#         mask = 1.0 - dones[t]
#         delta = rewards[t] + gamma * last_val * mask - values[t]
#         last_adv = delta + gamma * lam * last_adv * mask
#         advantages = advantages.write(t, last_adv)
#         last_val = values[t]

#     advantages = advantages.stack()

#     returns = advantages + values

#     advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

#     return advantages, returns

@tf.function
def train_step(p_model, v_model, p_optimizer, v_optimizer, board_batch, pieces_batch, actions_batch,
               old_log_probs, advantages_batch, returns_batch,
               old_values_batch, beta):
    input_actions = actions_batch[:, :-1]
    target_actions = actions_batch[:, 1:]
    mask = tf.cast(tf.logical_and(target_actions != 0, target_actions != 11), tf.float32) # START and PAD

    with tf.GradientTape() as p_tape:
        logits, piece_scores, key_scores = p_model((board_batch, pieces_batch, input_actions),
                                                   training=True, return_scores=True)

        dist = tfp.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(target_actions)

        # Stabilizer penalty
        entropy = tf.reduce_mean(tf.reduce_sum(dist.entropy(), axis=-1))
        approx_kl = tf.reduce_mean(tf.reduce_sum(old_log_probs - new_log_probs, axis=-1))
        
        # Policy loss
        ratio = tf.exp(new_log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - ppo_clip, 1 + ppo_clip)
        policy_loss = -tf.reduce_mean(tf.minimum(
            ratio * advantages_batch,
            clipped_ratio * advantages_batch
        ))
        
        # Compute metrics
        clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32))

        # Total policy loss
        total_policy_loss = policy_loss - entropy_coef * entropy + beta * approx_kl
    
    # Apply policy gradients
    p_gradients = p_tape.gradient(total_policy_loss, p_model.trainable_variables)
    p_optimizer.apply_gradients(zip(p_gradients, p_model.trainable_variables))
    
    with tf.GradientTape() as v_tape:
        values = v_model((board_batch, pieces_batch), training=True)
        values = tf.squeeze(values, axis=-1)

        # Value loss
        value_error = values - returns_batch
        clipped_values = old_values_batch + tf.clip_by_value(
            values - old_values_batch, -value_clip, value_clip
        )
        clipped_value_error = clipped_values - returns_batch
        value_loss = tf.reduce_mean(tf.maximum(
            tf.square(value_error),
            tf.square(clipped_value_error)
        ))
    
    # Apply value gradients
    v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
    v_optimizer.apply_gradients(zip(v_gradients, v_model.trainable_variables))

    return (policy_loss, value_loss, entropy, approx_kl,
            clipped_frac, board_batch, scores)


# def train_on_dataset(p_model, v_model, p_optimizer, v_optimizer, dataset, num_epochs, beta):
#     for batch in dataset.take(num_epochs):
#         step_out = train_step(p_model, v_model,
#                               p_optimizer, v_optimizer,
#                               batch['boards'],
#                               batch['pieces'],
#                               batch['actions'],
#                               batch['old_log_probs'],
#                               batch['advantages'],
#                               batch['returns'],
#                               batch['old_values'],
#                               beta)
#     return step_out

# def main(argv):
#     # Initialize adaptive KL term
#     beta = 1.0

#     # WandB logging config
#     config = {
#         'num_envs': num_envs,
#         'num_collection_steps': num_collection_steps,
#         'mini_batch_size': mini_batch_size,
#         'epochs_per_gen': epochs_per_gen,
#         'gamma': gamma,
#         'lam': lam,
#         'ppo_clip': ppo_clip,
#         'value_clip': value_clip,
#         'entropy_coef': entropy_coef,
#         'target_kl': target_kl,
#         'kl_tolerance': kl_tolerance,
#     }

#     # Initialize WandB logging
#     wandb_run = wandb.init(
#         project='Tetris',
#         # id='djunejcq',
#         # resume='must',
#         config=config,
#     )

#     # Set up environments
#     constructors = [lambda idx=i: PyTetrisEnv(queue_size=queue_size,
#                                               max_holes=max_holes,
#                                               seed=123,
#                                               idx=idx)
#                     for i in range(num_envs)]
#     ppy_env = ParallelPyEnvironment(constructors, start_serially=True, blocking=False)
#     tf_env = TFPyEnvironment(ppy_env)
#     last_time = time.time()
#     print("Initialized environments", flush=True)

#     (all_boards, all_pieces,
#      all_actions, all_log_probs,
#      all_values, all_attacks,
#      all_height_penalty, all_hole_penalty,
#      all_skyline_penalty, all_bumpy_penalty,
#      all_death_penalty, all_dones) = collect_trajectory(p_model,
#                                                         v_model,
#                                                         tf_env,
#                                                         num_collection_steps,
#                                                         num_envs,
#                                                         render=True)

    # # Collect trajectories and train
    # for gen in range(generations):

    #     # Collect trajectory
    #     print(f"{time.time() - last_time:2.2f} | Collecting trajectory...", flush=True)
    #     last_time = time.time()
        
    #     # Render every 10 generations
    #     (all_boards, all_pieces,
    #      all_actions, all_log_probs,
    #      all_values, all_attacks,
    #      all_height_penalty, all_hole_penalty,
    #      all_skyline_penalty, all_bumpy_penalty,
    #      all_death_penalty, all_dones) = collect_trajectory(p_model,
    #                                                         v_model,
    #                                                         tf_env,
    #                                                         num_collection_steps,
    #                                                         num_envs,
    #                                                         render=False)
    #     all_rewards = ((2 * all_attacks) ** 2 + all_height_penalty + all_hole_penalty +
    #                    all_skyline_penalty + all_bumpy_penalty + all_death_penalty + 
    #                    1.0) / 10.0

    #     print(f"{time.time() - last_time:2.2f} | Collected. Creating dataset...", flush=True)
    #     last_time = time.time()
    #     # Compute advantages and returns
    #     all_advantages, all_returns = compute_gae_and_returns(all_values,
    #                                                           all_rewards,
    #                                                           all_dones,
    #                                                           gamma, lam)
    #     # Flatten data
    #     boards_flat = tf.reshape(all_boards, (-1, 24, 10, 1))
    #     pieces_flat = tf.reshape(all_pieces, (-1, (queue_size + 2)))
    #     actions_flat = tf.reshape(all_actions, (-1,))
    #     log_probs_flat = tf.reshape(all_log_probs, (-1,))
    #     advantages_flat = tf.reshape(all_advantages, (-1,))
    #     returns_flat = tf.reshape(all_returns, (-1,))
    #     values_flat = tf.reshape(all_values, (-1,))
        
    #     # Create TF dataset from data
    #     dataset = (tf.data.Dataset.from_tensor_slices({'boards': boards_flat,
    #                                                    'pieces': pieces_flat,
    #                                                    'actions': actions_flat,
    #                                                    'old_log_probs': log_probs_flat,
    #                                                    'advantages': advantages_flat,
    #                                                    'returns': returns_flat,
    #                                                    'old_values': values_flat})
    #                .shuffle(buffer_size=boards_flat.shape[0])
    #                .batch(mini_batch_size,
    #                       num_parallel_calls=tf.data.AUTOTUNE,
    #                       deterministic=False,
    #                       drop_remainder=True)
    #                .prefetch(tf.data.AUTOTUNE))
        
    #     print(f"{time.time() - last_time:2.2f} | Dataset made. Training on dataset...", flush=True)
    #     last_time = time.time()
        
    #     # Train on collected data
    #     train_out = train_on_dataset(p_model, v_model, p_optimizer, v_optimizer,
    #                                  dataset, epochs_per_gen, beta)
        
    #     # Save checkpoint
    #     p_checkpoint_manager.save()
    #     v_checkpoint_manager.save()

    #     print(f"{time.time() - last_time:2.2f} | Trained on dataset. Logging metrics...", flush=True)
    #     last_time = time.time()
        
    #     # Unpack metrics
    #     (policy_loss, value_loss, entropy, approx_kl,
    #      clipped_frac, boards, scores) = train_out
        
    #     # Adjust beta
    #     if approx_kl > target_kl * (1 + kl_tolerance):
    #         beta = 1.5
    #     elif approx_kl < target_kl * (1 - kl_tolerance):
    #         beta = 1.0 / 1.5
    #     else:
    #         beta = 1.0
        
    #     # Compute more metrics
    #     avg_probs = tf.reduce_mean(tf.exp(all_log_probs))
    #     avg_reward = tf.reduce_mean(tf.reduce_sum(all_rewards, axis=0))
    #     avg_attacks = tf.reduce_mean(tf.reduce_sum(all_attacks, axis=0))
    #     avg_height_penalty = tf.reduce_mean(tf.reduce_sum(all_height_penalty, axis=0))
    #     avg_hole_penalty = tf.reduce_mean(tf.reduce_sum(all_hole_penalty, axis=0))
    #     avg_skyline_penalty = tf.reduce_mean(tf.reduce_sum(all_skyline_penalty, axis=0))
    #     avg_bumpy_penalty = tf.reduce_mean(tf.reduce_sum(all_bumpy_penalty, axis=0))
    #     avg_death_penalty = tf.reduce_mean(tf.reduce_sum(all_death_penalty, axis=0))
    #     avg_deaths = tf.reduce_mean(tf.reduce_sum(all_dones, axis=0))
    #     avg_pieces = tf.reduce_mean(num_collection_steps / (tf.reduce_sum(all_dones, axis=0) + 1))
        
    #     c_scores = tf.reshape(tf.reduce_mean(scores, axis=[0, 2, 3])[0], (12, 5, 1))
    #     norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (tf.reduce_max(c_scores) - tf.reduce_min(c_scores))
        
    #     wandb.log({'policy_loss': policy_loss,
    #                'value_loss': value_loss,
    #                'entropy': entropy,
    #                'beta': beta,
    #                'approx_kl': approx_kl,
    #                'clipped_frac': clipped_frac,
    #                'avg_probs': avg_probs,
    #                'avg_reward': avg_reward,
    #                'avg_attacks': avg_attacks,
    #                'avg_height_penalty': avg_height_penalty,
    #                'avg_hole_penalty': avg_hole_penalty,
    #                'avg_skyline_penalty': avg_skyline_penalty,
    #                'avg_bumpy_penalty': avg_bumpy_penalty,
    #                'avg_death_penalty': avg_death_penalty,
    #                'avg_deaths': avg_deaths,
    #                'avg_pieces': avg_pieces,
    #                'board': wandb.Image(boards[0, ..., 0]),
    #                'scores': wandb.Image(norm_c_scores)})
        
    #     print(f"{time.time() - last_time:2.2f} | Gen: {gen} | Reward: {avg_reward}", flush=True)
    #     last_time = time.time()
    
    # tf_env.close()
    # wandb_run.finish()

# if __name__ == '__main__':
#     tf_agents.system.multiprocessing.handle_main(main)
