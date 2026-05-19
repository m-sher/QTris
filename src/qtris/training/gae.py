"""GAE + raw-return reducers, shared across all PPO trainers.

Functions take `num_collection_steps` and `num_envs` as args (they were
module-level constants in each trainer before Phase 3). `tf.function` traces
per unique value, but in practice each training run uses one value.
"""
import tensorflow as tf


@tf.function(jit_compile=True)
def compute_gae_and_returns(
    values, last_values, rewards, dones, gamma, lam, num_collection_steps, num_envs
):
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
def compute_raw_returns(rewards, dones, gamma, num_collection_steps, num_envs):
    returns = tf.TensorArray(
        dtype=tf.float32, size=num_collection_steps, element_shape=(num_envs, 1)
    )

    last_ret = tf.zeros(returns.element_shape, dtype=tf.float32)

    for t in tf.range(num_collection_steps - 1, -1, -1):
        mask = 1.0 - dones[t]
        last_ret = rewards[t] + gamma * last_ret * mask
        returns = returns.write(t, last_ret)

    return tf.ensure_shape(returns.stack(), (num_collection_steps, num_envs, 1))
