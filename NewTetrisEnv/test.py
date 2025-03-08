import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_agents
import tensorflow_probability as tfp
from tf_agents.environments import suite_gym, ParallelPyEnvironment, TFPyEnvironment
import numpy as np
import pygame

# Hyperparameters
gamma = 0.99                # Discount factor
lam = 0.95                  # GAE lambda
epsilon_clip = 0.2          # Policy clipping parameter
value_clip = 0.5            # Value clipping parameter
entropy_coef = 0.01         # Entropy regularization coefficient
value_loss_coef = 0.5       # Value loss coefficient
num_epochs = 10             # Number of training epochs
mini_batch_size = 64        # Mini-batch size
num_envs = 32               # Number of parallel environments
max_steps_per_env = 500     # Steps to collect per environment
target_kl = 0.01            # Target KL divergence for adaptive KL
tolerance = 0.3             # Tolerance for KL adjustment
beta = 1.0                  # Initial KL penalty coefficient
num_iterations = 50         # Total training iterations

# if not tf_agents.system.default.multiprocessing_core._INITIALIZED[0]:
#     tf_agents.system.multiprocessing.enable_interactive_mode()

class PPOActorCriticNetwork(tf.keras.Model):
    def __init__(self):
        super(PPOActorCriticNetwork, self).__init__()
        # Shared layers
        self.shared_layers = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu')
        ], name='shared_layers')
        # Policy head: outputs logits for 3 actions
        self.policy_head = tf.keras.layers.Dense(3, activation=None)
        # Value head: outputs a single value
        self.value_head = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.shared_layers(inputs)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

def collect_data(model, env, max_steps_per_env, num_envs):
    """
    Collects data from multiple environments using preallocated NumPy arrays.

    Args:
        model: PPOActorCriticNetwork instance
        env: TFPyEnvironment with ParallelPyEnvironment
        max_steps_per_env: Number of steps to collect per environment
        num_envs: Number of parallel environments

    Returns:
        Tuple of NumPy arrays containing observations, actions, rewards,
        log probabilities, values, and next observations.
    """

    all_observations = tf.TensorArray(dtype=tf.float32, size=max_steps_per_env,
                                      element_shape=(num_envs, 6))
    all_actions = tf.TensorArray(dtype=tf.int32, size=max_steps_per_env,
                                 element_shape=(num_envs,))
    all_log_probs = tf.TensorArray(dtype=tf.float32, size=max_steps_per_env,
                                   element_shape=(num_envs,))
    all_values = tf.TensorArray(dtype=tf.float32, size=max_steps_per_env,
                                element_shape=(num_envs,))
    all_rewards = tf.TensorArray(dtype=tf.float32, size=max_steps_per_env,
                                 element_shape=(num_envs,))
    all_dones = tf.TensorArray(dtype=tf.float32, size=max_steps_per_env,
                               element_shape=(num_envs,))

    time_step = env.reset()
    observation = time_step.observation  # Shape: (num_envs, 4)

    for step in range(max_steps_per_env):
        # Compute actions and values
        logits, values = model(observation)
        dist = tfp.distributions.Categorical(logits=logits)
        actions = dist.sample()  # Shape: (num_envs,)
        log_probs = dist.log_prob(actions)  # Shape: (num_envs,)

        # Step the environment
        time_step = env.step(actions)
        rewards = time_step.reward
        done = time_step.is_last()

        all_observations = all_observations.write(step, observation)
        all_actions = all_actions.write(step, actions)
        all_log_probs = all_log_probs.write(step, log_probs)
        all_values = all_values.write(step, tf.squeeze(values, axis=-1))
        all_rewards = all_rewards.write(step, rewards)
        all_dones = all_dones.write(step, tf.cast(done, tf.float32))

        observation = time_step.observation

    all_observations = all_observations.stack()
    all_actions = all_actions.stack()
    all_log_probs = all_log_probs.stack()
    all_values = all_values.stack()
    all_rewards = all_rewards.stack()
    all_dones = all_dones.stack()

    return (all_observations, all_actions, all_log_probs,
            all_values, all_rewards, all_dones)

@tf.function
def compute_gae_and_returns(values, rewards, dones, gamma, lam):
    """
    Computes GAE and returns using TensorFlow operations.

    Args:
        values: Value predictions (max_steps_per_env, num_envs)
        rewards: Raw rewards received (max_steps_per_env, num_envs)
        dones: Episode terminations (max_steps_per_env, num_envs)
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: GAE advantages (max_steps_per_env, num_envs)
        returns: Returns (max_steps_per_env, num_envs)
    """

    advantages = tf.TensorArray(dtype=tf.float32, size=max_steps_per_env,
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
def train_step(model, optimizer, obs_batch, actions_batch, old_log_probs_batch,
               advantages_batch, returns_batch, old_values_batch, beta):
    """Performs a single training step."""
    with tf.GradientTape() as tape:
        logits, values = model(obs_batch)
        values = tf.squeeze(values, axis=-1)

        # Policy loss
        dist = tfp.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        ratio = tf.exp(log_probs - old_log_probs_batch)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon_clip, 1 + epsilon_clip)
        policy_loss = -tf.reduce_mean(tf.minimum(
            ratio * advantages_batch,
            clipped_ratio * advantages_batch
        ))

        # Value loss with clipping
        value_error = values - returns_batch
        clipped_values = old_values_batch + tf.clip_by_value(
            values - old_values_batch, -value_clip, value_clip
        )
        clipped_value_error = clipped_values - returns_batch
        value_loss = tf.reduce_mean(tf.maximum(
            tf.square(value_error),
            tf.square(clipped_value_error)
        ))

        # Entropy and KL
        entropy = tf.reduce_mean(dist.entropy())
        kl_estimate = tf.reduce_mean(old_log_probs_batch - log_probs)

        # Total loss
        total_loss = (policy_loss + value_loss_coef * value_loss -
                      entropy_coef * entropy + beta * kl_estimate)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return kl_estimate

def render_greedy_episode(model, max_steps=200):
    """
    Renders a greedy episode of the trained PPO model using Pygame.

    Args:
        model: PPOActorCriticNetwork instance
        max_steps: Maximum number of steps to render (default: 200)
    """
    pygame.init()

    # Create a single environment for rendering
    env = TFPyEnvironment(suite_gym.load("Acrobot-v1"))
    time_step = env.reset()

    # Get the initial frame to determine window size
    frame = env.render(mode='rgb_array')[0].numpy()  # Shape: (height, width, 3)
    height, width, _ = frame.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Acrobot-v1")

    # Initialize clock for frame rate control
    clock = pygame.time.Clock()

    # Main rendering loop
    for step in range(max_steps):
        # Get the observation and compute the greedy action
        obs = time_step.observation
        logits, _ = model(obs)
        action = tf.argmax(logits, axis=1)  # Shape: (1,)

        # Step the environment
        time_step = env.step(action)

        # Render the frame
        frame = env.render(mode='rgb_array')[0].numpy()  # Convert tensor to NumPy array
        # Transpose frame from (height, width, 3) to (width, height, 3) for Pygame
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(surface, (0, 0))  # Draw the surface on the screen
        pygame.display.flip()  # Update the display

        # Handle Pygame events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Control frame rate (10 FPS, matching the original 0.1s delay)
        clock.tick(30)

        # Exit if the episode ends
        if time_step.is_last():
            break

    # Clean up Pygame resources
    pygame.quit()

def train_on_dset(model, optimizer, dataset, num_epochs):
    # Training epochs
    kl_estimates = []
    for epoch in range(num_epochs):
        for batch in dataset:
            kl = train_step(model, optimizer,
                            batch['obs'], batch['actions'],
                            batch['old_log_probs'], batch['advantages'],
                            batch['returns'], batch['old_values'], beta)
            kl_estimates.append(kl.numpy())
    mean_kl = np.mean(kl_estimates)
    return mean_kl

def main(argv):

    # Initialize model and optimizer
    model = PPOActorCriticNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Set up environments
    env_constructors = [lambda: suite_gym.load("Acrobot-v1")
                        for _ in range(num_envs)]

    parallel_env = ParallelPyEnvironment(env_constructors)
    tf_env = TFPyEnvironment(parallel_env)

    for iteration in range(num_iterations):
        # Collect data
        (all_observations, all_actions, all_log_probs,
         all_values, all_rewards, all_dones) = collect_data(model, tf_env,
                                                            max_steps_per_env,
                                                            num_envs)
        # Compute GAE and returns
        all_advantages, all_returns = compute_gae_and_returns(all_values, all_rewards,
                                                              all_dones, gamma, lam)

        # Flatten data using reshape (no loops)
        obs_flat = tf.reshape(all_observations, (-1, 6))
        actions_flat = tf.reshape(all_actions, (-1,))
        log_probs_flat = tf.reshape(all_log_probs, (-1,))
        advantages_flat = tf.reshape(all_advantages, (-1,))
        returns_flat = tf.reshape(all_returns, (-1,))
        values_flat = tf.reshape(all_values, (-1,))

        # Create TensorFlow dataset
        dataset = (tf.data.Dataset.from_tensor_slices({
                       'obs': obs_flat,
                       'actions': actions_flat,
                       'old_log_probs': log_probs_flat,
                       'advantages': advantages_flat,
                       'returns': returns_flat,
                       'old_values': values_flat})
                   .shuffle(buffer_size=obs_flat.shape[0])
                   .batch(mini_batch_size,
                          num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False,
                          drop_remainder=True)
                   .prefetch(tf.data.AUTOTUNE))

        avg_reward = np.mean(np.sum(all_rewards, axis=0))

        mean_kl = train_on_dset(model, optimizer, dataset, num_epochs)

        # Adjust beta
        if mean_kl > target_kl * (1 + tolerance):
            beta = 1.5
        elif mean_kl < target_kl * (1 - tolerance):
            beta = 1.0 / 1.5
        else:
            beta = 1.0

        # Print progress
        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, "
                  f"Reward: {avg_reward:3.2f}, Mean KL: {mean_kl:.4f}, "
                  f"Beta: {beta:.4f}")

    input("Press Enter to render an episode...")
    render_greedy_episode(model)

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(main)
  exit()