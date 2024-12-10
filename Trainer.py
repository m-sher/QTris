import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import wandb
from Player import Player
from tf_agents.replay_buffers import TFUniformReplayBuffer

class Trainer():
    def __init__(self, agent, critic, ref_model, max_len, gamma, lam, temperature=1.0, max_episode_steps=2000, render=True):
        self.eps = 1e-10
        self.ppo_epsilon = 0.2
        self.agent = agent
        self.critic = critic
        self.ref_model = ref_model
        self.gamma = gamma
        self.lam = lam
        self.temp = temperature
        self.player = Player(max_len=max_len)
        self.max_episode_steps = max_episode_steps
        self.wandb_run = wandb.init(
            project='Tetris'
        )
        
        if render:
            fig, ax = plt.subplots()
            img = ax.imshow(tf.zeros((28, 10)), vmin=0, vmax=1)
            self.renderer = (fig, img)
        else:
            self.renderer = None

    @tf.function(reduce_retracing=True)
    def _compute_gae(self, values, rewards, gamma, lam):
        advantages = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
        last_adv = tf.constant([0.0])
        last_val = values[-1]

        last_ind = tf.shape(rewards)[0] - 1
        for t in tf.range(last_ind, -1, -1):
            if t == last_ind and rewards[t] == -1:
                last_val = tf.constant([0.0])
            
            
            delta = rewards[t] + gamma * last_val - values[t]
            last_adv = delta + gamma * lam * last_adv
            advantages = advantages.write(t, last_adv)
            last_val = values[t]

        advantages = advantages.stack()
        
        returns = values + advantages
        
        return advantages, returns
    
    @tf.function
    def _ppo_loss_fn(self, valid_mask, new_probs, old_probs, advantages):

        # valid_mask -> batch, max_len, 1
        # new_probs -> batch, max_len, 1
        # old_probs -> batch, max_len, 1
        # advantages -> batch, 1

        # batch, 1, 1
        advantages = ((advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + self.eps))[..., None]
        
        ratio = tf.exp(new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

        unclipped_proportion = tf.reduce_sum(tf.cast(ratio == clipped_ratio, tf.float32) * valid_mask) / tf.reduce_sum(valid_mask)

        # batch, max_len, 1
        clipped = clipped_ratio * advantages
        unclipped = ratio * advantages

        ppo_loss = -tf.reduce_sum(tf.minimum(clipped, unclipped) * valid_mask) / tf.reduce_sum(valid_mask)
        
        return ppo_loss, unclipped_proportion

    @tf.function
    def _critic_loss_fn(self, valid_mask, returns, values):
        # valid_mask -> batch, max_len, 1
        # returns -> batch, 1
        # values -> batch, max_len, 1
        
        # batch, max_len, 1
        raw_loss = (returns[..., None] - values) ** 2

        critic_loss = tf.reduce_sum(raw_loss * valid_mask) / tf.reduce_sum(valid_mask)
        
        return critic_loss
    
    @tf.function
    def _train_step(self, board_batch, piece_batch, input_batch, old_probs, advantages, returns, training_actor):

        # batch, max_len
        inputs = input_batch[:, :-1]
        actions = input_batch[:, 1:]
        
        # batch, max_len, 1
        valid_mask = tf.cast(actions != 0, tf.float32)[..., None]
        
        with tf.GradientTape() as critic_tape:
            values = self.critic((board_batch, piece_batch, inputs), training=True)
    
            critic_loss = self._critic_loss_fn(valid_mask, returns, values)

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        if training_actor:
            with tf.GradientTape() as agent_tape:
                logits, piece_scores, key_scores = self.agent((board_batch, piece_batch, inputs), training=True, return_scores=True)
                ref_logits, ref_piece_scores, ref_key_scores = self.ref_model((board_batch, piece_batch, inputs), training=False, return_scores=True)
                
                scores = {'current': [piece_scores, key_scores],
                          'reference': [ref_piece_scores, ref_key_scores]}
                
                # batch, max_len, key_dim
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                ref_log_probs = tf.nn.log_softmax(ref_logits, axis=-1)
                
                # batch, max_len, 1
                new_probs = tf.gather(log_probs,
                                      actions,
                                      batch_dims=2)[..., None]

                entropy = tf.reduce_sum(log_probs * tf.exp(log_probs) * valid_mask) / tf.reduce_sum(valid_mask)
                
                ppo_loss, unclipped_proportion = self._ppo_loss_fn(valid_mask, new_probs, old_probs, advantages)

                raw_kl_div = keras.losses.KLDivergence(reduction='none')(tf.exp(ref_log_probs), tf.exp(log_probs))
                kl_div = tf.reduce_sum(raw_kl_div[..., None] * valid_mask) / tf.reduce_sum(valid_mask)
    
                agent_loss = ppo_loss + 0.01 * entropy # + 0.01 * kl_div

            agent_grads = agent_tape.gradient(agent_loss, self.agent.trainable_variables)
            self.agent.optimizer.apply_gradients(zip(agent_grads, self.agent.trainable_variables))

            return ppo_loss, entropy, kl_div, critic_loss, unclipped_proportion, scores
        
        return critic_loss

    def train(self, gens, train_steps=100, training_actor=False):
        
        for gen in range(gens):
            
            # Run episode
            episode_data = self.player.run_episode(self.agent, self.critic, max_steps=self.max_episode_steps,
                                                   greedy=False, temperature=self.temp, renderer=self.renderer)
            episode_boards, episode_pieces, episode_inputs, episode_probs, episode_values, episode_rewards = episode_data
            episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self.gamma, self.lam)

            # Print metrics
            avg_reward = tf.reduce_mean(episode_rewards)
            sum_reward = tf.reduce_sum(episode_rewards)
            
            print(f'\rCurrent Gen: {gen + 1}\t|\tAvg Reward: {avg_reward:1.1f}\t|\tTotal Reward: {sum_reward:1.1f}\t|', end='', flush=True)

            dset = (tf.data.Dataset.from_tensor_slices((episode_boards, episode_pieces, episode_inputs, episode_probs, episode_advantages, episode_returns))
                    .shuffle(self.max_episode_steps)
                    .repeat()
                    .batch(128,
                           num_parallel_calls=tf.data.AUTOTUNE,
                           deterministic=False,
                           drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))
            
            for i, (board_batch, piece_batch, input_batch,
                    prob_batch, advantage_batch, return_batch) in enumerate(dset.take(train_steps)):
                
                step_out = self._train_step(board_batch, piece_batch, input_batch,
                                            prob_batch, advantage_batch, return_batch, training_actor)
            
            if training_actor:
                ppo_loss, entropy, kl_div, critic_loss, unclipped_proportion, scores = step_out
                print(f'\rPPO Loss: {ppo_loss:1.2f}\t|\tKL Divergence: {kl_div:1.2f}\t|\tCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)

                c_scores = tf.reshape(tf.reduce_mean(scores['current'][0], axis=[0, 2, 3])[0], (14, 5, 1))
                norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (tf.reduce_max(c_scores) - tf.reduce_min(c_scores))
                
                r_scores = tf.reshape(tf.reduce_mean(scores['reference'][0], axis=[0, 2, 3])[0], (14, 5, 1))
                norm_r_scores = (r_scores - tf.reduce_min(r_scores)) / (tf.reduce_max(r_scores) - tf.reduce_min(r_scores))

                score_diff = (c_scores - r_scores) ** 2
                norm_score_diff = (score_diff - tf.reduce_min(score_diff)) / (tf.reduce_max(score_diff) - tf.reduce_min(score_diff))
                
                wandb.log({'ppo_loss': ppo_loss,
                           'entropy': entropy,
                           'kl_div': kl_div,
                           'unclipped_proportion': unclipped_proportion,
                           'critic_loss': critic_loss,
                           'reward': sum_reward,
                           'reward_per_piece': avg_reward,
                           'board': wandb.Image(board_batch[0]),
                           'current_scores': wandb.Image(norm_c_scores),
                           'reference_scores': wandb.Image(norm_r_scores),
                           'score_diff': wandb.Image(norm_score_diff)})
            else:
                critic_loss = step_out
                print(f'\rCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)
                
                wandb.log({'critic_loss': critic_loss,
                           'reward': sum_reward,
                           'reward_per_piece': avg_reward})