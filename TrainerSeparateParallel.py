import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
from tensorflow import keras
import wandb
from Pretrainer import Pretrainer
from PlayerSeparateParallel import Player


class Trainer():
    def __init__(self, actor, critic, max_len, num_players, players_to_render, gamma, lam, exp_coef=0.01, temperature=1.0, max_holes=4, max_episode_steps=1000):
        self._eps = 1e-10
        self._ppo_epsilon = 0.2
        self.actor = actor
        self.critic = critic
        self._num_players = num_players
        self._gamma = gamma
        self._lam = lam
        self._temp = temperature
        self._exp_coef = exp_coef
        self._reward_eps = 0.01
        self._pretrainer = Pretrainer(gamma, max_len)
        self._pretrainer.cache_dset()
        self._exp_dset = self._pretrainer.gt_dset
        self._kl = keras.losses.KLDivergence()
        self._scc = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                               reduction='none')
        
        self.player = Player(max_len=max_len,
                             num_players=num_players,
                             players_to_render=players_to_render,
                             max_holes=max_holes,
                             reward_eps=self._reward_eps)
        
        self._max_episode_steps = max_episode_steps
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
            if t == last_ind:
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
        advantages = ((advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + self._eps))
        
        ratio = tf.exp(new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self._ppo_epsilon, 1 + self._ppo_epsilon)

        clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32))

        # batch, max_len, 1
        clipped = clipped_ratio * advantages
        unclipped = ratio * advantages

        ppo_loss = -tf.reduce_mean(tf.minimum(clipped, unclipped))
        
        return ppo_loss, clipped_frac

    @tf.function
    def _exp_loss_fn(self, exp_target_batch, exp_logits):
        # exp_target_batch -> batch, max_len
        # exp_logits -> batch, max_len, key_dim
        
        exp_valid_mask = tf.cast(exp_target_batch != 0, tf.float32)
        
        raw_loss = self._scc(exp_target_batch, exp_logits)
        
        exp_loss = tf.reduce_sum(raw_loss * exp_valid_mask) / tf.reduce_sum(exp_valid_mask)
        
        return exp_loss

    @tf.function
    def _critic_loss_fn(self, returns, values, nov_valid_mask):
        # valid_mask -> batch, max_len, 1
        # returns -> batch, 1
        # values -> batch, max_len, 1
        
        # batch, max_len, 1
        raw_loss = (returns[..., None] - values) ** 2

        critic_loss = tf.reduce_sum(raw_loss * nov_valid_mask) / tf.reduce_sum(nov_valid_mask)
        
        return critic_loss
    
    @tf.function
    def _critic_step(self, board_batch, piece_batch, input_batch, returns, nov_valid_mask):
        
        with tf.GradientTape() as critic_tape:
            values = self.critic((board_batch, piece_batch, input_batch), training=True)
    
            critic_loss = self._critic_loss_fn(returns, values, nov_valid_mask)

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return critic_loss
    
    @tf.function
    def _actor_step(self, nov_board_batch, nov_piece_batch, nov_input_batch, nov_action_batch,
                    prob_batch, advantage_batch, nov_valid_mask, exp_board_batch, exp_piece_batch,
                    exp_input_batch, exp_target_batch):
        
        with tf.GradientTape() as actor_tape:
            nov_logits, piece_scores, key_scores = self.actor((nov_board_batch, nov_piece_batch, nov_input_batch), training=True, return_scores=True)
            
            scores = (piece_scores, key_scores)
            
            # batch, max_len, key_dim
            log_probs = tf.nn.log_softmax(nov_logits, axis=-1)

            # batch,
            action_ind = tf.cast(tf.reduce_sum(nov_valid_mask, axis=[1, 2]) - 1, tf.int32)
            
            # batch, key_dim
            action_probs = tf.gather(log_probs,
                                     action_ind,
                                     batch_dims=1)
            
            entropy = tf.reduce_mean(action_probs * tf.exp(action_probs))
            
            # batch, 1
            new_probs = tf.gather(action_probs,
                                  nov_action_batch,
                                  batch_dims=1)[..., None]
            
            old_probs = tf.gather(prob_batch,
                                  nov_action_batch,
                                  batch_dims=1)[..., None]
            
            ppo_loss, clipped_frac = self._ppo_loss_fn(new_probs, old_probs, advantage_batch)
            
            exp_logits, _, _ = self.actor((exp_board_batch, exp_piece_batch, exp_input_batch), training=True, return_scores=True)
            
            exp_loss = self._exp_loss_fn(exp_target_batch, exp_logits)
            
            entropy_coef = tf.exp(tf.reduce_mean(new_probs))
            avg_probs = tf.reduce_mean(old_probs)

            kl_div = self._kl(tf.exp(prob_batch), tf.exp(action_probs))

            actor_loss = ppo_loss + entropy_coef * entropy + self._exp_coef * exp_loss

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
        return ppo_loss, entropy_coef, entropy, exp_loss, avg_probs, kl_div, clipped_frac, scores

    def _update_step(self, nov_dset):
        
        for i, (nov_batch, exp_batch) in enumerate(tf.data.Dataset.zip((nov_dset, self._exp_dset))):
            
            (nov_board_batch, nov_piece_batch, nov_input_batch, nov_action_batch,
             prob_batch, advantage_batch, return_batch) = nov_batch
            
            (exp_board_batch, exp_piece_batch, exp_input_batch, exp_target_batch, _) = exp_batch
            
            nov_valid_mask = tf.cast(nov_input_batch != 0, tf.float32)[..., None]
            
            critic_step_out = self._critic_step(nov_board_batch, nov_piece_batch, nov_input_batch,
                                                return_batch, nov_valid_mask)
            critic_loss = critic_step_out
            
            actor_step_out = self._actor_step(nov_board_batch, nov_piece_batch, nov_input_batch, nov_action_batch,
                                              prob_batch, advantage_batch, nov_valid_mask, exp_board_batch,
                                              exp_piece_batch, exp_input_batch, exp_target_batch)
            ppo_loss, entropy_coef, entropy, exp_loss, avg_probs, kl_div, clipped_frac, scores = actor_step_out
            
        return (critic_loss, ppo_loss, entropy_coef, entropy, exp_loss,
                avg_probs, kl_div, clipped_frac, scores[0], nov_board_batch[0])

    def train(self, gens, update_steps=4):
        
        for gen in range(gens):
            
            # Run episode
            episode_data = self.player.run_episode(self.actor, self.critic, max_steps=self._max_episode_steps,
                                                   greedy=False, temperature=self._temp)
            (all_episode_boards, all_episode_pieces, all_episode_inputs,
             all_episode_actions, all_episode_probs, all_episode_values, all_episode_rewards) = episode_data

            all_episode_advantages, all_episode_returns = [], []
            for episode_values, episode_rewards in zip(all_episode_values, all_episode_rewards):
                episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self._gamma, self._lam)
                all_episode_advantages.append(episode_advantages)
                all_episode_returns.append(episode_returns)

            # Compute metrics
            avg_reward, sum_reward, avg_deaths, avg_pieces = [], [], [], []
            
            # COMBINE WITH ABOVE WHY ARE WE LOOPING OVER REWARDS TWICE
            for episode_rewards in all_episode_rewards:
                num_pieces = tf.reduce_sum(tf.cast(episode_rewards != -self._reward_eps, tf.float32))
                total_reward = tf.reduce_sum(episode_rewards)
                
                avg_reward.append(total_reward / num_pieces)
                sum_reward.append(total_reward)
                avg_deaths.append(tf.cast(episode_rewards[-1] == -1, tf.float32))
                avg_pieces.append(num_pieces)
            
            avg_reward = tf.reduce_mean(avg_reward)
            sum_reward = tf.reduce_mean(sum_reward)
            avg_deaths = tf.reduce_mean(avg_deaths)
            avg_pieces = tf.reduce_mean(avg_pieces)

            all_episode_boards = tf.concat(all_episode_boards, axis=0)
            all_episode_pieces = tf.concat(all_episode_pieces, axis=0) 
            all_episode_inputs = tf.concat(all_episode_inputs, axis=0)
            all_episode_actions = tf.concat(all_episode_actions, axis=0)
            all_episode_probs = tf.concat(all_episode_probs, axis=0)
            all_episode_advantages = tf.concat(all_episode_advantages, axis=0)
            all_episode_returns = tf.concat(all_episode_returns, axis=0)
            
            print(f'\rCurrent Gen: {gen + 1:4d}\t|\tAvg Reward: {avg_reward:1.1f}\t|\tTotal Reward: {sum_reward:1.1f}\t|', end='', flush=True)

            nov_dset = (tf.data.Dataset.from_tensor_slices((all_episode_boards, all_episode_pieces, all_episode_inputs, all_episode_actions,
                                                            all_episode_probs, all_episode_advantages, all_episode_returns))
                        .shuffle(self._max_episode_steps * self._num_players)
                        .batch(512,
                               num_parallel_calls=tf.data.AUTOTUNE,
                               deterministic=False,
                               drop_remainder=True)
                        .prefetch(tf.data.AUTOTUNE))
            
            print('\rMade Dataset', end='', flush=True)
            
            for i in range(update_steps):
                step_out = self._update_step(nov_dset)
                print(f'\rUpdate Step {i}', end='', flush=True)
            
            (critic_loss, ppo_loss, entropy_coef, entropy, exp_loss,
             avg_probs, kl_div, clipped_frac, scores, board_example) = step_out

            print(f'\rPPO Loss: {ppo_loss:1.2f}\t|\tKL Divergence: {kl_div:1.2f}\t|\tCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)

            c_scores = tf.reshape(tf.reduce_mean(scores, axis=[0, 2, 3])[0], (14, 5, 1))
            norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (tf.reduce_max(c_scores) - tf.reduce_min(c_scores))
            
            wandb.log({'ppo_loss': ppo_loss,
                       'entropy_coef': entropy_coef,
                       'entropy': entropy,
                       'exp_loss': exp_loss,
                       'avg_probs': avg_probs,
                       'kl_div': kl_div,
                       'clipped_frac': clipped_frac,
                       'critic_loss': critic_loss,
                       'reward': sum_reward,
                       'reward_per_piece': avg_reward,
                       'avg_deaths': avg_deaths,
                       'avg_pieces_placed': avg_pieces,
                       'board': wandb.Image(board_example),
                       'current_scores': wandb.Image(norm_c_scores)})

    # FIX THIS 
    def save_demo(self, filename, max_steps):
        # Open piece display array
        with open('PieceDisplay.npy', 'rb') as f:
            piece_array = np.load(f)

        # Run episode greedily
        episode_data = self.player.run_episode(self.actor, self.critic, max_steps=self._max_episode_steps,
                                               greedy=True, temperature=self._temp)
        (all_episode_boards, all_episode_pieces, all_episode_inputs,
         all_episode_actions, all_episode_probs, all_episode_values, all_episode_rewards) = episode_data
        
        best_episode_ind = tf.argmax([tf.reduce_sum(episode_rewards) for episode_rewards in all_episode_rewards])
        best_episode_boards = all_episode_boards[best_episode_ind]
        best_episode_pieces = all_episode_pieces[best_episode_ind]
        
        # Generate frames from episode_date
        frames = []
        for board, pieces in zip(best_episode_boards, best_episode_pieces):
            board_frame = Image.fromarray((board[..., 0] * 255).numpy().astype(np.uint8)).resize((100, 280), Image.Resampling.NEAREST)
            piece_frame = Image.fromarray(np.concatenate([(piece_array[piece] * 255).astype(np.uint8)
                                                           for piece in pieces], axis=0)).resize((50, 280), Image.Resampling.NEAREST)
            frames.append(Image.fromarray(np.concatenate([np.array(board_frame), np.array(piece_frame)], axis=1)))

         # Write frames to file
        imageio.mimsave(filename, frames, duration=0.5)