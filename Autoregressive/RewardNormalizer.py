import tensorflow as tf


class RewardNormalizer:
    def __init__(self, num_envs, gamma=0.99, initial_var=1.0, decay=0.99, epsilon=1e-8, clip=10.0):
        self.num_envs = num_envs
        self.gamma = gamma
        self.decay = decay
        self.epsilon = epsilon
        self.clip = clip

        self.return_var = initial_var
        self.env_returns = tf.Variable(
            tf.zeros((num_envs, 1), dtype=tf.float32), trainable=False
        )

    def _compute_discounted_returns(self, rewards, dones):
        num_steps = tf.shape(rewards)[0]
        all_returns = tf.TensorArray(dtype=tf.float32, size=num_steps, element_shape=(self.num_envs, 1))

        ret = self.env_returns.value()

        for t in tf.range(num_steps):
            ret = ret * self.gamma * (1.0 - dones[t]) + rewards[t]
            all_returns = all_returns.write(t, ret)

        self.env_returns.assign(ret)

        return all_returns.stack()

    def update(self, rewards, dones):
        discounted_returns = self._compute_discounted_returns(rewards, dones)
        batch_var = tf.math.reduce_variance(discounted_returns)
        self.return_var = self.decay * self.return_var + (1.0 - self.decay) * batch_var

    def normalize(self, rewards):
        scaled = rewards / (tf.sqrt(self.return_var) + self.epsilon)
        return tf.clip_by_value(scaled, -self.clip, self.clip)

    @property
    def std(self):
        return tf.sqrt(self.return_var)
