import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import wandb
from Player import Player
from tf_agents.replay_buffers import TFUniformReplayBuffer

class Trainer():
    def __init__(self, model, ref_model, optimizer, max_len, gamma, lam, max_episode_steps=2000, buffer_cap=10000):
        self.eps = 1e-10
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.player = Player(max_len=max_len)
        self.max_episode_steps = max_episode_steps
        self.wandb_run = wandb.init(
            project='Tetris'
        )
        
        data_spec = (tf.TensorSpec(shape=(28, 10, 1), dtype=tf.float32, name='Boards'),
                     tf.TensorSpec(shape=(7,), dtype=tf.int32, name='Pieces'),
                     tf.TensorSpec(shape=(max_len+1,), dtype=tf.int32, name='Input'),
                     tf.TensorSpec(shape=(max_len, 1), dtype=tf.float32, name='Probs'),
                     tf.TensorSpec(shape=(1,), dtype=tf.float32, name='Advantage'),
                     tf.TensorSpec(shape=(1,), dtype=tf.float32, name='Return'))

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec,
            batch_size=1,
            max_length=buffer_cap
        )

        fig, ax = plt.subplots()
        img = ax.imshow(tf.zeros((28, 10)), vmin=0, vmax=1)
        self.renderer = (fig, img)

    @tf.function(reduce_retracing=True)
    def _compute_gae(self, values, rewards, gamma, lam):
        advantages = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
        gae = tf.constant([0.0])

        last_ind = tf.shape(rewards)[0] - 1
        for t in tf.range(last_ind, -1, -1):
            delta = rewards[t] + gamma * (values[t + 1] - values[t]) if t != last_ind else tf.constant([0.0])
            gae = delta + gamma * lam * gae
            advantages = advantages.write(t, gae)

        advantages = advantages.stack()
        
        returns = values + advantages
        
        return advantages, returns
    
    def fill_replay_buffer(self, max_episodes=50):
        for i in range(max_episodes):
            if self.replay_buffer.num_frames() >= self.replay_buffer.capacity:
                break
        
            episode_data = self.player.run_episode(self.model, max_steps=self.max_episode_steps, greedy=False, renderer=self.renderer)
            episode_boards, episode_pieces, episode_inputs, episode_probs, episode_values, episode_rewards = episode_data
            episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self.gamma, self.lam)

            # Add data to replay buffer
            for frame in zip(episode_boards, episode_pieces, episode_inputs,
                             episode_probs, episode_advantages, episode_returns):
                board, pieces, inputs, probs, adv, ret = frame
                self.replay_buffer.add_batch((board[None, ...],
                                              pieces[None, ...],
                                              inputs[None, ...],
                                              probs[None, ...],
                                              adv[None, ...],
                                              ret[None, ...]))
                
            print(f'\rCurrent Episode: {i}', end='', flush=True)
        print('\rDone filling replay buffer', end='', flush=True)

    @tf.function
    def _ppo_loss_fn(self, valid_mask, new_probs, old_probs, advantages):

        # valid_mask -> batch, max_len, 1
        # new_probs -> batch, max_len, 1
        # old_probs -> batch, max_len, 1
        # advantages -> batch, 1
        
        epsilon = 0.2

        # batch, 1, 1
        advantages = ((advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + self.eps))[:, None, :]
        
        ratio = tf.exp(new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)

        unclipped_proportion = tf.reduce_sum(tf.cast(ratio == clipped_ratio, tf.float32) * valid_mask) / tf.reduce_sum(valid_mask)

        # batch, max_len, 1
        clipped = clipped_ratio * advantages * valid_mask
        unclipped = ratio * advantages * valid_mask
    
        ppo_loss = -tf.reduce_sum(tf.minimum(clipped, unclipped)) / tf.reduce_sum(valid_mask)
        
        return ppo_loss, unclipped_proportion

    @tf.function
    def _critic_loss_fn(self, valid_mask, returns, values):
        # valid_mask -> batch, max_len, 1
        # returns -> batch, 1
        # values -> batch, max_len, 1

        masked_diff = (returns[:, None, :] - values) * valid_mask
        critic_loss = tf.reduce_sum(masked_diff ** 2) / tf.reduce_sum(valid_mask)
        return critic_loss
    
    @tf.function
    def _ppo_train_step(self, board_batch, piece_batch, input_batch, old_probs, advantages, returns):

        with tf.GradientTape() as tape:
            inputs = input_batch[:, :-1]
            actions = input_batch[:, 1:]
            
            logits, values = self.model((board_batch, piece_batch, inputs), training=True)
            ref_logits, _ = self.ref_model((board_batch, piece_batch, inputs), training=False)

            # batch, max_len, key_dim
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            ref_log_probs = tf.nn.log_softmax(ref_logits, axis=-1)

            # batch, max_len, 1
            new_probs = tf.gather(log_probs,
                                  actions,
                                  batch_dims=2)[..., None]

            # batch, max_len, 1
            valid_mask = tf.cast(actions != 0, tf.float32)[..., None]
            
            ppo_loss, unclipped_proportion = self._ppo_loss_fn(valid_mask, new_probs, old_probs, advantages)
            
            kl_penalty = keras.losses.KLDivergence()(tf.exp(ref_log_probs), tf.exp(log_probs))

            actor_loss = ppo_loss # + kl_penalty
            critic_loss = self._critic_loss_fn(valid_mask, returns, values)
            loss = actor_loss + critic_loss
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return ppo_loss, kl_penalty, unclipped_proportion, critic_loss
    
    def train(self, gens, train_steps=100):
        
        for gen in range(gens):
            
            # Run episode
            episode_data = self.player.run_episode(self.model, max_steps=self.max_episode_steps, greedy=False, renderer=self.renderer)
            episode_boards, episode_pieces, episode_inputs, episode_probs, episode_values, episode_rewards = episode_data
            episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self.gamma, self.lam)

            # Add data to replay buffer
            for frame in zip(episode_boards, episode_pieces, episode_inputs,
                             episode_probs, episode_advantages, episode_returns):
                board, pieces, inputs, probs, adv, ret = frame
                self.replay_buffer.add_batch((board[None, ...],
                                              pieces[None, ...],
                                              inputs[None, ...],
                                              probs[None, ...],
                                              adv[None, ...],
                                              ret[None, ...]))

            # Print metrics
            avg_reward = tf.reduce_mean(episode_rewards)
            sum_reward = tf.reduce_sum(episode_rewards)
            
            print(f'\rCurrent Gen: {gen + 1}\t|\tAvg Reward: {avg_reward:1.1f}\t|\tTotal Reward: {sum_reward:1.1f}\t|', end='', flush=True)
        
            # Make dataset sampling from replay buffer
            dset = self.replay_buffer.as_dataset(
                sample_batch_size=128,
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
            
            for i, ((board_batch, piece_batch, input_batch,
                     prob_batch, advantage_batch, return_batch), _) in enumerate(dset.take(train_steps)):
                
                losses = self._ppo_train_step(board_batch, piece_batch, input_batch, 
                                              prob_batch, advantage_batch, return_batch)

            ppo_loss, kl_penalty, unclipped_proportion, critic_loss = losses
            print(f'\rPPO Loss: {ppo_loss:1.2f}\t|\tKL Penalty: {kl_penalty:1.2f}\t|\tCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)
            wandb.log({'ppo_loss': ppo_loss,
                       'kl_penalty': kl_penalty,
                       'unclipped_proportion': unclipped_proportion,
                       'critic_loss': critic_loss,
                       'reward': sum_reward,
                       'reward_per_piece': avg_reward})