import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from Player import Player
from tf_agents.replay_buffers import TFUniformReplayBuffer

class Trainer():
    def __init__(self, model, optimizers, seq_len, gamma, lam, max_episode_steps=2000, buffer_cap=10000):
        self.eps = 1e-10
        self.model = model
        self.actor_optimizer, self.critic_optimizer = optimizers
        self.gamma = gamma
        self.lam = lam
        self.player = Player(max_len=seq_len)
        self.max_episode_steps = max_episode_steps
        self.wandb_run = wandb.init(
            project='Tetris'
        )
        
        data_spec = (tf.TensorSpec(shape=(28, 10, 1), dtype=tf.float32, name='Boards'),
                     tf.TensorSpec(shape=(7,), dtype=tf.int32, name='Pieces'),
                     tf.TensorSpec(shape=(seq_len,), dtype=tf.int32, name='ChosenAction'),
                     tf.TensorSpec(shape=(seq_len-1,), dtype=tf.float32, name='ActionProbs'),
                     tf.TensorSpec(shape=(1,), dtype=tf.float32, name='Advantages'),
                     tf.TensorSpec(shape=(1,), dtype=tf.float32, name='Returns'))

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

        for t in tf.range(tf.shape(rewards)[0])[::-1]:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages = advantages.write(t, gae)

        advantages = advantages.stack()
        returns = values[:-1] + advantages

        return advantages, returns
    
    def fill_replay_buffer(self, max_episodes=50):
        for i in range(max_episodes):
            if self.replay_buffer.num_frames() >= self.replay_buffer.capacity:
                break
        
            episode_data = self.player.run_episode(self.model, max_steps=self.max_episode_steps, greedy=False, renderer=self.renderer)
            episode_boards, episode_pieces, episode_actions, episode_probs, episode_values, episode_rewards = episode_data
            episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self.gamma, self.lam)
            
            for frame in zip(episode_boards, episode_pieces, episode_actions, episode_probs, episode_advantages, episode_returns):
                board, pieces, action, probs, adv, ret = frame
                self.replay_buffer.add_batch((board[None, ...], pieces[None, ...], action[None, ...], probs[None, ...], adv[None, ...], ret[None, ...]))
            print(f'\rCurrent Episode: {i}', end='', flush=True)
        print('\rDone filling replay buffer', end='\n', flush=True)

    @tf.function
    def _actor_loss_fn(self, mask, new_probs, old_probs, advantages):
        
        advantages = ((advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + self.eps))
        
        ratio = tf.exp(new_probs - old_probs)
    
        clipped = tf.clip_by_value(ratio, 0.9, 1.1) * advantages * mask
        unclipped = ratio * advantages * mask
    
        ppo_loss = -tf.reduce_sum(tf.minimum(clipped, unclipped)) / tf.reduce_sum(mask)
        
        return ppo_loss

    @tf.function
    def _critic_loss_fn(self, returns, values):
        critic_loss = tf.reduce_sum((returns - values) ** 2) / tf.cast(tf.shape(returns)[0], tf.float32)
        return critic_loss
    
    @tf.function
    def _ppo_train_step(self, board_batch, piece_batch, action_batch, old_probs, advantages, returns, training_actor):
        
        board_rep, _ = self.model.process_board((board_batch, piece_batch), training=False)
        
        with tf.GradientTape() as critic_tape:
            values, _ = self.model.process_vals(board_rep, training=True)

            critic_loss = self._critic_loss_fn(returns, values)
        
        critic_grads = critic_tape.gradient(critic_loss, self.critic_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_variables))

        if training_actor:
            with tf.GradientTape() as actor_tape:
                logits, _ = self.model.process_keys((board_rep, action_batch[..., :-1]), training=True)
                action_probs = tf.nn.log_softmax(logits, axis=-1)
            
                mask = tf.cast(action_batch[..., 1:] != 0, tf.float32)
                
                new_probs = tf.gather(action_probs,
                                      action_batch[..., 1:],
                                      batch_dims=2)
                
                actor_loss = self._actor_loss_fn(mask, new_probs, old_probs, advantages)

            actor_grads = actor_tape.gradient(actor_loss, self.actor_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_variables))
        
            return actor_loss, critic_loss
        else:
            return critic_loss
    
    def train(self, gens, train_steps=100, training_actor=False):
        self.critic_variables = []
        self.actor_variables = []
        for layer in self.model.layers:
            if 'venc' in layer.name or layer.name == 'critic_top':
                layer.trainable = True
                for var in layer.trainable_variables:
                    self.critic_variables.append(var)
            elif training_actor and ('kdec' in layer.name or layer.name == 'actor_top'):
                layer.trainable = True
                for var in layer.trainable_variables:
                    self.actor_variables.append(var)
            else:
                layer.trainable = False
        
        for gen in range(gens):
            # Run episode
            episode_data = self.player.run_episode(self.model, max_steps=self.max_episode_steps, greedy=False, renderer=self.renderer)
            episode_boards, episode_pieces, episode_actions, episode_probs, episode_values, episode_rewards = episode_data
            episode_advantages, episode_returns = self._compute_gae(episode_values, episode_rewards, self.gamma, self.lam)
            
            for frame in zip(episode_boards, episode_pieces, episode_actions, episode_probs, episode_advantages, episode_returns):
                board, pieces, action, probs, adv, ret = frame
                self.replay_buffer.add_batch((board[None, ...], pieces[None, ...], action[None, ...], probs[None, ...], adv[None, ...], ret[None, ...]))
        
            avg_reward = tf.reduce_mean(episode_rewards)
            sum_reward = tf.reduce_sum(episode_rewards)
            
            print(f'\rCurrent Gen: {gen + 1}\t|\tAvg Reward: {avg_reward:1.1f}\t|\tTotal Reward: {sum_reward:1.1f}\t|', end='\n', flush=True)
        
            # Make dataset sampling from replay buffer
            dset = self.replay_buffer.as_dataset(
                sample_batch_size=128,
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
            
            for i, ((board_batch, piece_batch, action_batch, old_probs, advantage_batch, return_batch), _) in enumerate(dset.take(train_steps)):
                losses = self._ppo_train_step(board_batch, piece_batch, action_batch, old_probs, advantage_batch, return_batch, training_actor)

            if training_actor:
                actor_loss, critic_loss = losses
                print(f'\rActor Loss: {actor_loss:1.2f}\t|\tCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)
                wandb.log({'actor_loss': actor_loss,
                           'critic_loss': critic_loss,
                           'reward': sum_reward,
                           'reward_per_piece': avg_reward})
            else:
                critic_loss = losses
                print(f'\rCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)
                wandb.log({'critic_loss': critic_loss,
                           'reward': sum_reward,
                           'reward_per_piece': avg_reward})