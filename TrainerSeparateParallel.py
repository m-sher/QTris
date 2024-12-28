import tensorflow as tf
from tensorflow import keras
import wandb
from PlayerSeparateParallel import Player

class Trainer():
    def __init__(self, actor, critic, ref_model, max_len, num_players, gamma, lam, temperature=1.0, max_episode_steps=2000):
        self.eps = 1e-10
        self.ppo_epsilon = 0.2
        self.actor = actor
        self.critic = critic
        self.ref_model = ref_model
        self.num_players = num_players
        self.gamma = gamma
        self.lam = lam
        self.temp = temperature
        self.player = Player(max_len=max_len, num_players=num_players)
        self.max_episode_steps = max_episode_steps
        self.wandb_run = wandb.init(
            project='Tetris'
        )

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
    def _ppo_loss_fn(self, new_probs, old_probs, advantages):

        # new_probs -> batch, 1
        # old_probs -> batch, 1
        # advantages -> batch, 1

        # batch, 1
        advantages = ((advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + self.eps))
        
        ratio = tf.exp(new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

        clipped_frac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.ppo_epsilon, tf.float32))

        # batch, max_len, 1
        clipped = clipped_ratio * advantages
        unclipped = ratio * advantages

        ppo_loss = -tf.reduce_mean(tf.minimum(clipped, unclipped))
        
        return ppo_loss, clipped_frac

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
    def _critic_step(self, board_batch, piece_batch, input_batch, returns, valid_mask):
        
        with tf.GradientTape() as critic_tape:
            values = self.critic((board_batch, piece_batch, input_batch), training=True)
    
            critic_loss = self._critic_loss_fn(valid_mask, returns, values)

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return critic_loss
    
    @tf.function
    def _actor_step(self, board_batch, piece_batch, input_batch, action_batch, old_probs, advantages, valid_mask):
        
        with tf.GradientTape() as actor_tape:
            logits, piece_scores, key_scores = self.actor((board_batch, piece_batch, input_batch), training=True, return_scores=True)
            ref_logits, ref_piece_scores, ref_key_scores = self.ref_model((board_batch, piece_batch, input_batch), training=False, return_scores=True)
            
            scores = {'current': [piece_scores, key_scores],
                      'reference': [ref_piece_scores, ref_key_scores]}
            
            # batch, max_len, key_dim
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            ref_log_probs = tf.nn.log_softmax(ref_logits, axis=-1)

            # batch,
            action_ind = tf.cast(tf.reduce_sum(valid_mask, axis=[1, 2]) - 1, tf.int32)
            
            # batch, key_dim
            action_probs = tf.gather(log_probs,
                                     action_ind,
                                     batch_dims=1)
            
            ref_action_probs = tf.gather(ref_log_probs,
                                         action_ind,
                                         batch_dims=1)
            
            entropy = tf.reduce_mean(action_probs * tf.exp(action_probs))
            
            # batch, 1
            new_probs = tf.gather(action_probs,
                                  action_batch,
                                  batch_dims=1)[..., None]
            
            ppo_loss, clipped_frac = self._ppo_loss_fn(new_probs, old_probs, advantages)

            kl_div = keras.losses.KLDivergence()(tf.exp(ref_action_probs), tf.exp(action_probs))

            actor_loss = ppo_loss + 0.001 * entropy

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
        return ppo_loss, entropy, kl_div, clipped_frac, scores

    def _update_step(self, dset):
        
        for i, (board_batch, piece_batch, input_batch, action_batch,
                prob_batch, advantage_batch, return_batch) in enumerate(dset):
            
            valid_mask = tf.cast(input_batch != 0, tf.float32)[..., None]
            
            critic_step_out = self._critic_step(board_batch, piece_batch, input_batch,
                                                return_batch, valid_mask)
            critic_loss = critic_step_out
            
            actor_step_out = self._actor_step(board_batch, piece_batch, input_batch, action_batch,
                                              prob_batch, advantage_batch, valid_mask)
            ppo_loss, entropy, kl_div, clipped_frac, scores = actor_step_out
                  
        return critic_loss, ppo_loss, entropy, kl_div, scores, clipped_frac, board_batch[0]

    def train(self, gens, update_steps=4):
        
        for gen in range(gens):
            
            # Run episode
            episode_data = self.player.run_episode(self.actor, self.critic, max_steps=self.max_episode_steps,
                                                   greedy=False, temperature=self.temp)
            (all_episode_boards, all_episode_pieces, all_episode_inputs,
             all_episode_actions, all_episode_probs, all_episode_values, all_episode_rewards) = episode_data

            all_episode_advantages, all_episode_returns = [], []
            for episode_values, episode_rewards in zip(all_episode_values, all_episode_rewards):
                episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self.gamma, self.lam)
                all_episode_advantages.append(episode_advantages)
                all_episode_returns.append(episode_returns)

            all_episode_boards = tf.concat(all_episode_boards, axis=0)
            all_episode_pieces = tf.concat(all_episode_pieces, axis=0)
            all_episode_inputs = tf.concat(all_episode_inputs, axis=0)
            all_episode_actions = tf.concat(all_episode_actions, axis=0)
            all_episode_probs = tf.concat(all_episode_probs, axis=0)
            all_episode_advantages = tf.concat(all_episode_advantages, axis=0)
            all_episode_returns = tf.concat(all_episode_returns, axis=0)
            
            # Print metrics
            avg_reward = tf.reduce_mean([tf.reduce_mean(episode_rewards) for episode_rewards in all_episode_rewards])
            sum_reward = tf.reduce_mean([tf.reduce_sum(episode_rewards) for episode_rewards in all_episode_rewards])
            
            print(f'\rCurrent Gen: {gen + 1}\t|\tAvg Reward: {avg_reward:1.1f}\t|\tTotal Reward: {sum_reward:1.1f}\t|', end='', flush=True)

            dset = (tf.data.Dataset.from_tensor_slices((all_episode_boards, all_episode_pieces, all_episode_inputs, all_episode_actions,
                                                        all_episode_probs, all_episode_advantages, all_episode_returns))
                    .shuffle(self.max_episode_steps * self.num_players)
                    .batch(512,
                           num_parallel_calls=tf.data.AUTOTUNE,
                           deterministic=False,
                           drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))
            
            for _ in range(update_steps):
                step_out = self._update_step(dset)
            
            critic_loss, ppo_loss, entropy, kl_div, scores, clipped_frac, board_example = step_out
                
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
                       'clipped_frac': clipped_frac,
                       'critic_loss': critic_loss,
                       'reward': sum_reward,
                       'reward_per_piece': avg_reward,
                       'board': wandb.Image(board_example),
                       'current_scores': wandb.Image(norm_c_scores),
                       'reference_scores': wandb.Image(norm_r_scores),
                       'score_diff': wandb.Image(norm_score_diff)})

    # FIX THIS 
    # def save_demo(self, filename, max_steps):

    #     # Open piece display array
    #     with open('PieceDisplay.npy', 'rb') as f:
    #         piece_array = np.load(f)

    #     # Run episode greedily
    #     episode_data = self.player.run_episode(self.actor, self.critic, max_steps=max_steps,
    #                                            greedy=True, temperature=self.temp, renderer=self.renderer)
    #     episode_boards, episode_pieces, episode_inputs, episode_actions, episode_probs, episode_values, episode_rewards = episode_data

    #     # Generate frames from episode_date
    #     frames = []
    #     for board, pieces in zip(episode_boards, episode_pieces):
    #         board_frame = Image.fromarray((board[..., 0] * 255).numpy().astype(np.uint8)).resize((100, 280), Image.Resampling.NEAREST)
    #         piece_frame = Image.fromarray(np.concatenate([(piece_array[piece] * 255).astype(np.uint8)
    #                                                       for piece in pieces], axis=0)).resize((50, 280), Image.Resampling.NEAREST)
    #         frames.append(Image.fromarray(np.concatenate([np.array(board_frame), np.array(piece_frame)], axis=1)))

    #     # Write frames to file
    #     imageio.mimsave(filename, frames, duration=0.5)