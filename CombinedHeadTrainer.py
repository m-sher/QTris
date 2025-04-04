from TFTetrisEnv.PyTetrisEnv import PyTetrisEnv
from CombinedHeadTetrisModel import TetrisModel
from TFTetrisEnv.Moves import Convert
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tf_agents
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
import pygame
import wandb
import time

# Environment params
generations = 10000
num_envs = 4
num_collection_steps = 500
queue_size = 5
max_holes = None

# Training params
mini_batch_size = 1024
epochs_per_gen = 10

gamma = 0.99
lam = 0.95
ppo_clip = 0.2
value_clip = 0.2

value_coef = 0.5
entropy_coef = 0.07

target_kl = 0.01
kl_tolerance = 0.3

def collect_trajectory(model, env, num_collection_steps, num_envs, render):
    all_boards = tf.TensorArray(dtype=tf.int32, size=num_collection_steps,
                                      element_shape=(num_envs, 24, 10))
    all_pieces = tf.TensorArray(dtype=tf.int32, size=num_collection_steps,
                                      element_shape=(num_envs, queue_size + 2))
    all_actions = tf.TensorArray(dtype=tf.int32, size=num_collection_steps,
                                 element_shape=(num_envs,))
    all_log_probs = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                   element_shape=(num_envs,))
    all_values = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                element_shape=(num_envs,))
    all_attacks = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                 element_shape=(num_envs,))
    all_height_penalty = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                        element_shape=(num_envs,))
    all_hole_penalty = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                      element_shape=(num_envs,))
    all_bumpy_penalty = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                       element_shape=(num_envs,))
    all_death_penalty = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                       element_shape=(num_envs,))
    all_dones = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                               element_shape=(num_envs,))

    # Initialize pygame
    if render:
        pygame.init()
        screen = pygame.display.set_mode((250, 600))
        pygame.display.set_caption("Tetris")

    time_step = env.reset()
    observation = time_step.observation # dict of batched boards and pieces

    for step in range(num_collection_steps):
        board = observation['board']
        pieces = observation['pieces']

        # Render the state of the first environment
        if render:
            screen.fill((0, 0, 0))
            board_surf = pygame.Surface((10, 24))
            pygame.surfarray.blit_array(board_surf, board[0].T * 255)
            board_surf = pygame.transform.scale(board_surf, (250, 600))
            screen.blit(board_surf, (0, 0))
            pygame.display.update()

        # Run model prediction
        logits, values = model((board, pieces), training=False)
        dist = tfp.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        # Convert actions to dict
        converted_actions = tf.gather(Convert.to_move,
                                      actions)

        action_dict = {
            'hold': converted_actions[:, 0],
            'standard': converted_actions[:, 1],
            'spin': converted_actions[:, 2],
        }

        time_step = env.step(action_dict)
        reward = time_step.reward
        attack = reward['attack']
        height_reward = reward['height_penalty']
        hole_penalty = reward['hole_penalty']
        bumpy_penalty = reward['bumpy_penalty']
        death_penalty = reward['death_penalty']

        dones = time_step.is_last()

        dones = tf.cast(dones, tf.float32)

        all_boards = all_boards.write(step, board)
        all_pieces = all_pieces.write(step, pieces)
        all_actions = all_actions.write(step, actions)
        all_log_probs = all_log_probs.write(step, log_probs)
        all_values = all_values.write(step, values)
        all_attacks = all_attacks.write(step, attack)
        all_height_penalty = all_height_penalty.write(step, height_reward)
        all_hole_penalty = all_hole_penalty.write(step, hole_penalty)
        all_bumpy_penalty = all_bumpy_penalty.write(step, bumpy_penalty)
        all_death_penalty = all_death_penalty.write(step, death_penalty)
        all_dones = all_dones.write(step, dones)

        observation = time_step.observation

    all_boards = all_boards.stack()
    all_pieces = all_pieces.stack()
    all_actions = all_actions.stack()
    all_log_probs = all_log_probs.stack()
    all_values = all_values.stack()
    all_attacks = all_attacks.stack()
    all_height_penalty = all_height_penalty.stack()
    all_hole_penalty = all_hole_penalty.stack()
    all_bumpy_penalty = all_bumpy_penalty.stack()
    all_death_penalty = all_death_penalty.stack()
    all_dones = all_dones.stack()

    return (all_boards, all_pieces, all_actions, all_log_probs,
            all_values, all_attacks, all_height_penalty,
            all_hole_penalty, all_bumpy_penalty, all_death_penalty,
            all_dones)

@tf.function
def compute_gae_and_returns(values, rewards, dones, gamma, lam):
    """
    Args:
        values: Value predictions (num_collection_steps, num_envs)
        rewards: Raw rewards received (num_collection_steps, num_envs)
        dones: Episode terminations (num_collection_steps, num_envs)
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: GAE advantages (num_collection_steps, num_envs)
        returns: Returns (num_collection_steps, num_envs)
    """

    advantages = tf.TensorArray(dtype=tf.float32, size=num_collection_steps,
                                element_shape=(num_envs,))

    last_adv = tf.zeros(advantages.element_shape, dtype=tf.float32)
    last_val = values[-1]

    for t in tf.range(advantages.size() - 1, -1, -1):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_val * mask - values[t]
        last_adv = delta + gamma * lam * last_adv * mask
        advantages = advantages.write(t, last_adv)
        last_val = values[t]

    advantages = advantages.stack()

    returns = advantages + values

    advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    return advantages, returns

@tf.function
def train_step(model, optimizer, board_batch, pieces_batch, actions_batch,
               old_log_probs, advantages_batch, returns_batch,
               old_values_batch, beta):
    with tf.GradientTape() as tape:
        logits, values, scores = model((board_batch, pieces_batch),
                                       training=True, return_scores=True)
        dist = tfp.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_batch)

        # Stabilizer penalty
        entropy = tf.reduce_mean(dist.entropy())
        approx_kl = tf.reduce_mean(old_log_probs - new_log_probs)
        
        # Policy loss
        ratio = tf.exp(new_log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - ppo_clip, 1 + ppo_clip)
        policy_loss = -tf.reduce_mean(tf.minimum(
            ratio * advantages_batch,
            clipped_ratio * advantages_batch
        ))
        
        # Compute metrics
        clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32))
        
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

        # Total loss
        total_loss = (policy_loss + value_coef * value_loss -
                      entropy_coef * entropy + beta * approx_kl)
    
    # Apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return (policy_loss, value_loss, entropy, approx_kl,
            clipped_frac, board_batch, scores)


def train_on_dataset(model, optimizer, dataset, num_epochs, beta):
    for _ in range(num_epochs):
        for batch in dataset:
            step_out = train_step(model, optimizer,
                                  batch['boards'],
                                  batch['pieces'],
                                  batch['actions'],
                                  batch['old_log_probs'],
                                  batch['advantages'],
                                  batch['returns'],
                                  batch['old_values'],
                                  beta)
    return step_out

def main(argv):
    # Initialize adaptive KL term
    beta = 1.0
    
    # Model params
    piece_dim = 8
    depth = 32
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.1
    trunk_dim = 128
    num_actions = Convert.to_move.shape[0]

    # WandB logging config
    config = {
        'num_envs': num_envs,
        'num_collection_steps': num_collection_steps,
        'mini_batch_size': mini_batch_size,
        'epochs_per_gen': epochs_per_gen,
        'gamma': gamma,
        'lam': lam,
        'ppo_clip': ppo_clip,
        'value_clip': value_clip,
        'value_coef': value_coef,
        'entropy_coef': entropy_coef,
        'target_kl': target_kl,
        'kl_tolerance': kl_tolerance,
    }

    # Initialize WandB logging
    wandb_run = wandb.init(
        project='Tetris',
        # id='w5nwaqvm',
        # resume='must',
        config=config,
    )
    
    # Initialize model and optimizer
    model = TetrisModel(piece_dim=piece_dim,
                        depth=depth,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate,
                        trunk_dim=trunk_dim,
                        num_actions=num_actions)
    optimizer = keras.optimizers.Adam(3e-4)
    model.compile(optimizer=optimizer)
    print("Initialized model and optimizer", flush=True)

    # Initialize checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, './combined_train_checkpoints', max_to_keep=3)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    # checkpoint_manager = tf.train.CheckpointManager(checkpoint, './combined_train_checkpoints', max_to_keep=3)
    print("Restored checkpoint", flush=True)

    # Set up environments
    constructors = [lambda idx=i: PyTetrisEnv(queue_size=queue_size,
                                              max_holes=max_holes,
                                              seed=123,
                                              idx=idx)
                    for i in range(num_envs)]
    ppy_env = ParallelPyEnvironment(constructors, start_serially=True, blocking=False)
    # tf_env = TFPyEnvironment(ppy_env)
    last_time = time.time()
    print("Initialized environments", flush=True)

    # Collect trajectories and train
    for gen in range(generations):
        # Collect trajectory
        print(f"{time.time() - last_time:2.2f} | Collecting trajectory...", flush=True)
        last_time = time.time()
        
        # Render every 10 generations
        (all_boards, all_pieces,
         all_actions, all_log_probs,
         all_values, all_attacks,
         all_height_penalty, all_hole_penalty,
         all_bumpy_penalty, all_death_penalty,
         all_dones) = collect_trajectory(model,
                                         ppy_env,
                                         num_collection_steps,
                                         num_envs,
                                         render=gen % 5 == 0)
        all_rewards = all_attacks + all_height_penalty + all_hole_penalty + all_bumpy_penalty + all_death_penalty
        # Standardize rewards
        all_norm_rewards = (all_rewards - tf.reduce_mean(all_rewards)) / (tf.math.reduce_std(all_rewards) + 1e-8)

        print(f"{time.time() - last_time:2.2f} | Collected. Creating dataset...", flush=True)
        last_time = time.time()
        # Compute advantages and returns
        all_advantages, all_returns = compute_gae_and_returns(all_values,
                                                              all_norm_rewards,
                                                              all_dones,
                                                              gamma, lam)
        # Flatten data
        boards_flat = tf.reshape(all_boards, (-1, 24, 10))
        pieces_flat = tf.reshape(all_pieces, (-1, (queue_size + 2)))
        actions_flat = tf.reshape(all_actions, (-1,))
        log_probs_flat = tf.reshape(all_log_probs, (-1,))
        advantages_flat = tf.reshape(all_advantages, (-1,))
        returns_flat = tf.reshape(all_returns, (-1,))
        values_flat = tf.reshape(all_values, (-1,))
        
        # Create TF dataset from data
        dataset = (tf.data.Dataset.from_tensor_slices({'boards': boards_flat,
                                                       'pieces': pieces_flat,
                                                       'actions': actions_flat,
                                                       'old_log_probs': log_probs_flat,
                                                       'advantages': advantages_flat,
                                                       'returns': returns_flat,
                                                       'old_values': values_flat})
                   .shuffle(buffer_size=boards_flat.shape[0])
                   .batch(mini_batch_size,
                          num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False,
                          drop_remainder=True)
                   .prefetch(tf.data.AUTOTUNE))
        
        print(f"{time.time() - last_time:2.2f} | Dataset made. Training on dataset...", flush=True)
        last_time = time.time()
        
        # Train on collected data
        train_out = train_on_dataset(model, optimizer, dataset,
                                     epochs_per_gen, beta)
        
        # Save checkpoint
        checkpoint_manager.save()

        print(f"{time.time() - last_time:2.2f} | Trained on dataset. Logging metrics...", flush=True)
        last_time = time.time()
        
        # Unpack metrics
        (policy_loss, value_loss, entropy, approx_kl,
         clipped_frac, boards, scores) = train_out
        
        # Adjust beta
        if approx_kl > target_kl * (1 + kl_tolerance):
            beta = 1.5
        elif approx_kl < target_kl * (1 - kl_tolerance):
            beta = 1.0 / 1.5
        else:
            beta = 1.0
        
        # Compute more metrics
        avg_probs = tf.reduce_mean(tf.exp(all_log_probs))
        avg_reward = tf.reduce_mean(tf.reduce_sum(all_rewards, axis=0))
        avg_attacks = tf.reduce_mean(tf.reduce_sum(all_attacks, axis=0))
        avg_height_penalty = tf.reduce_mean(tf.reduce_sum(all_height_penalty, axis=0))
        avg_hole_penalty = tf.reduce_mean(tf.reduce_sum(all_hole_penalty, axis=0))
        avg_bumpy_penalty = tf.reduce_mean(tf.reduce_sum(all_bumpy_penalty, axis=0))
        avg_death_penalty = tf.reduce_mean(tf.reduce_sum(all_death_penalty, axis=0))
        avg_deaths = tf.reduce_mean(tf.reduce_sum(all_dones, axis=0))
        avg_pieces = tf.reduce_mean(num_collection_steps / (tf.reduce_sum(all_dones, axis=0) + 1))
        
        c_scores = tf.reshape(tf.reduce_mean(scores, axis=[0, 2, 3])[0], (12, 5, 1))
        norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (tf.reduce_max(c_scores) - tf.reduce_min(c_scores))
        
        wandb.log({'policy_loss': policy_loss,
                   'value_loss': value_loss,
                   'entropy': entropy,
                   'beta': beta,
                   'approx_kl': approx_kl,
                   'clipped_frac': clipped_frac,
                   'avg_probs': avg_probs,
                   'avg_reward': avg_reward,
                   'avg_attacks': avg_attacks,
                   'avg_height_penalty': avg_height_penalty,
                   'avg_hole_penalty': avg_hole_penalty,
                   'avg_bumpy_penalty': avg_bumpy_penalty,
                   'avg_death_penalty': avg_death_penalty,
                   'avg_deaths': avg_deaths,
                   'avg_pieces': avg_pieces,
                   'board': wandb.Image(boards[0]),
                   'scores': wandb.Image(norm_c_scores)})
        
        print(f"{time.time() - last_time:2.2f} | Gen: {gen} | Reward: {avg_reward}", flush=True)
        last_time = time.time()
    
    ppy_env.close()
    wandb_run.finish()

if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(main)