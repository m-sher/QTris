import tensorflow as tf
import tensorflow_probability as tfp
import wandb
from PlayerSeparateParallel import Player


class Trainer():
    def __init__(self, model, ind_to_str, entropy_coef, entropy_decay, value_coef, num_players, players_to_render, gamma, lam, ckpt_type='pretrained', temperature=1.0, max_holes=4, max_height=10, max_diff=10, max_episode_steps=1000):
        self._eps = 1e-10
        self._ppo_epsilon = 0.2
        self._value_epsilon = 0.2
        self._entropy_coef = entropy_coef
        self._entropy_decay = entropy_decay
        self._value_coef = value_coef
        self.model = model
        self._num_players = num_players
        self._gamma = gamma
        self._lam = lam
        self._temp = temperature
                
        if ckpt_type == 'pretrained':
            checkpoint = tf.train.Checkpoint(model=model)
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'combined_checkpoints/pretrained', max_to_keep=5)
            checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print(f'Loaded checkpoint {self.checkpoint_manager.latest_checkpoint}')
            checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'combined_checkpoints/finetuned', max_to_keep=5)
        elif ckpt_type == 'finetuned':
            checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'combined_checkpoints/finetuned', max_to_keep=5)
            checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Loaded checkpoint {self.checkpoint_manager.latest_checkpoint}')
        else:
            checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
            self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'combined_checkpoints/finetuned', max_to_keep=5)

        self.player = Player(ind_to_str=ind_to_str,
                             num_players=num_players,
                             players_to_render=players_to_render,
                             max_holes=max_holes,
                             max_height=max_height,
                             max_diff=max_diff)
        
        self._max_episode_steps = max_episode_steps
        self.wandb_run = wandb.init(
            project='Tetris'
        )

    @tf.function(reduce_retracing=True)
    def _compute_gae(self, values, rewards, dones, gamma, lam):
        values = tf.ensure_shape(values, (None,))
        rewards = tf.ensure_shape(rewards, (None,))
        dones = tf.ensure_shape(dones, (None,))
        advantages = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
        last_adv = tf.constant(0.0)
        last_val = values[-1]

        last_ind = tf.shape(rewards)[0] - 1
        for t in tf.range(last_ind, -1, -1):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * last_val * mask - values[t]
            last_adv = delta + gamma * lam * last_adv * mask
            advantages = advantages.write(t, last_adv)
            last_val = values[t]

        advantages = advantages.stack()
        
        returns = tf.ensure_shape(values + advantages, (None,))
        
        return advantages, returns
    
    @tf.function
    def _ppo_loss_fn(self, new_probs, old_probs, advantages):

        # new_probs -> batch,
        # old_probs -> batch,
        # advantages -> batch,
        
        new_probs = tf.ensure_shape(new_probs, (None,))
        old_probs = tf.ensure_shape(old_probs, (None,))
        advantages = tf.ensure_shape(advantages, (None,))
        
        ratio = tf.exp(new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self._ppo_epsilon, 1 + self._ppo_epsilon)

        clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32))

        # batch, 1
        clipped = clipped_ratio * advantages
        unclipped = ratio * advantages

        ppo_loss = -tf.reduce_mean(tf.minimum(clipped, unclipped))
        
        return ppo_loss, clipped_frac

    @tf.function
    def _critic_loss_fn(self, returns, new_values, old_values):
        # returns -> batch, 1
        # new_values -> batch, 1
        # old_values -> batch, 1
        
        returns = tf.ensure_shape(returns, (None,))
        new_values = tf.ensure_shape(new_values, (None,))
        old_values = tf.ensure_shape(old_values, (None,))
        
        value_diff = new_values - old_values
        
        clipped_values = old_values + tf.clip_by_value(value_diff, -self._value_epsilon, self._value_epsilon)

        unclipped_loss = (new_values - returns) ** 2
        clipped_loss = (clipped_values - returns) ** 2

        critic_loss = tf.reduce_mean(tf.maximum(unclipped_loss, clipped_loss))
        
        return critic_loss
    
    @tf.function
    def _train_step(self, batch):
        
        (board_batch, piece_batch, action_batch,
         prob_batch, value_batch, advantage_batch, return_batch) = batch
        
        with tf.GradientTape() as tape:
            _, all_logits, all_values, scores = self.model(board_batch, piece_batch, training=True, return_scores=True)
            
            old_probs = prob_batch
            new_probs = tf.zeros_like(prob_batch, tf.float32) # batch,
            entropy = tf.zeros_like(prob_batch, tf.float32) # batch,
            for i, head_logits in enumerate(all_logits):
                # num_players,
                head_dist = tfp.distributions.Categorical(logits=head_logits / self._temp)
                
                new_probs += head_dist.log_prob(action_batch[:, i])
                
                entropy += head_dist.entropy()
            
            entropy = tf.reduce_mean(entropy)
            
            approx_kl = tf.reduce_mean(old_probs - new_probs)
            
            ppo_loss, clipped_frac = self._ppo_loss_fn(new_probs, old_probs, advantage_batch)
            
            avg_probs = tf.reduce_mean(old_probs)

            critic_loss = self._critic_loss_fn(return_batch, all_values, value_batch)

            total_loss = ppo_loss - self._entropy_coef * entropy + self._value_coef * critic_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
        return entropy, ppo_loss, clipped_frac, avg_probs, approx_kl, critic_loss, scores, board_batch

    def _run_epoch(self, dset):
        
        for batch in dset:
            
            step_out = self._train_step(batch)
        
        return step_out
        
    def train(self, gens, update_steps=4):
        
        for gen in range(gens):
            
            # Run episode
            episode_data = self.player.run_episode(self.model, max_steps=self._max_episode_steps,
                                                   greedy=False, temperature=self._temp)
            (all_episode_boards, all_episode_pieces, all_episode_actions,
             all_episode_probs, all_episode_values, all_episode_rewards, all_episode_dones) = episode_data

            all_episode_advantages, all_episode_returns = [], []
            
            sum_reward, avg_reward, avg_pieces, sum_deaths = [], [], [], []
            for episode_values, episode_rewards, episode_dones in zip(all_episode_values, all_episode_rewards, all_episode_dones):
                episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, episode_dones, self._gamma, self._lam)
                all_episode_advantages.append(episode_advantages)
                all_episode_returns.append(episode_returns)
                
                deaths = tf.reduce_sum(episode_dones)
                pieces = tf.reduce_sum(1.0 - episode_dones)
                sum_reward.append(tf.reduce_sum(episode_rewards) / (deaths + 1.0))
                avg_reward.append(tf.reduce_sum(episode_rewards) / (pieces + 1.0))
                avg_pieces.append(pieces / (deaths + 1.0))
                sum_deaths.append(deaths)
                
            sum_reward = tf.reduce_mean(sum_reward)
            avg_reward = tf.reduce_mean(avg_reward)
            sum_deaths = tf.reduce_mean(sum_deaths)
            avg_pieces = tf.reduce_mean(avg_pieces)

            all_episode_boards = tf.concat(all_episode_boards, axis=0)
            all_episode_pieces = tf.concat(all_episode_pieces, axis=0)
            all_episode_actions = tf.concat(all_episode_actions, axis=0)
            all_episode_probs = tf.concat(all_episode_probs, axis=0)
            all_episode_values = tf.concat(all_episode_values, axis=0)
            all_episode_returns = tf.concat(all_episode_returns, axis=0)

            all_episode_advantages = tf.concat(all_episode_advantages, axis=0)
            all_episode_advantages = (((all_episode_advantages - tf.reduce_mean(all_episode_advantages)) /
                                       (tf.math.reduce_std(all_episode_advantages) + self._eps)))
            
            print(f'\rCurrent Gen: {gen + 1:5d}\t|\tTotal Reward: {sum_reward:1.1f}\t|', end='', flush=True)

            dset = (tf.data.Dataset.from_tensor_slices((all_episode_boards, all_episode_pieces, all_episode_actions,
                                                        all_episode_probs, all_episode_values, all_episode_advantages,
                                                        all_episode_returns))
                    .shuffle(self._max_episode_steps * self._num_players)
                    .batch(512,
                           num_parallel_calls=tf.data.AUTOTUNE,
                           deterministic=False,
                           drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))
            
            print('\rMade Dataset', end='', flush=True)
            
            for i in range(update_steps):
                step_out = self._run_epoch(dset)
                print(f'\rUpdate Step {i}', end='', flush=True)
        
            entropy, ppo_loss, clipped_frac, avg_probs, approx_kl, critic_loss, scores, board_examples = step_out

            c_scores = tf.reshape(tf.reduce_mean(scores, axis=[0, 2, 3])[0], (14, 5, 1))
            norm_c_scores = (c_scores - tf.reduce_min(c_scores)) / (tf.reduce_max(c_scores) - tf.reduce_min(c_scores))
            
            wandb.log({'ppo_loss': ppo_loss,
                       'entropy_coef': self._entropy_coef,
                       'value_coef': self._value_coef,
                       'entropy': entropy,
                       'avg_probs': avg_probs,
                       'clipped_frac': clipped_frac,
                       'approx_kl': approx_kl,
                       'critic_loss': critic_loss,
                       'sum_reward': sum_reward,
                       'avg_reward': avg_reward,
                       'sum_deaths': sum_deaths,
                       'avg_pieces': avg_pieces,
                       'board': wandb.Image(board_examples[0]),
                       'current_scores': wandb.Image(norm_c_scores)})

            if (gen + 1) % 100 == 0:
                self._entropy_coef = max(self._entropy_coef * self._entropy_decay, 0.01)
                self.checkpoint_manager.save()

    # FIX THIS 
    """
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
    """