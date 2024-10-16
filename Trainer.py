import tensorflow as tf
import matplotlib.pyplot as plt
from Player import Player
from tf_agents.replay_buffers import TFUniformReplayBuffer

class Trainer():
    def __init__(self, model, optimizers, max_episode_steps=2000, buffer_cap=10000):
        self.model = model
        self.actor_optimizer, self.critic_optimizer = optimizers
        self.player = Player()
        self.max_episode_steps = max_episode_steps
        self.wandb_run = wandb.init(
            project='Tetris'
        )
        
        data_spec = (tf.TensorSpec(shape=(28, 10, 1), dtype=tf.float32, name='Boards'),
             tf.TensorSpec(shape=(7), dtype=tf.int32, name='Pieces'),
             tf.TensorSpec(shape=(max_len,), dtype=tf.int32, name='ChosenAction'),
             tf.TensorSpec(shape=(max_len-1, 1), dtype=tf.float32, name='ActionProbs'),
             tf.TensorSpec(shape=(1), dtype=tf.float32, name='Returns'),
             tf.TensorSpec(shape=(1), dtype=tf.float32, name='Advantage'))

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec,
            batch_size=1,
            max_length=buffer_cap
        )

        fig, ax = plt.subplots()
        img = ax.imshow(tf.zeros((28, 10)), vmin=0, vmax=1)
        self.renderer = (fig, img)
    
    def fill_replay_buffer(self, max_episodes=50, max_episode_steps=self.max_episode_steps):
        for i in range(max_episodes):
            if replay_buffer.num_frames() >= replay_buffer.capacity:
                break
        
            episode_data = self.player.run_episode(self.model, max_steps=max_episode_steps, greedy=False, renderer=self.renderer)
            episode_boards, episode_pieces, episode_actions, episode_probs, episode_rewards, episode_returns, _, episode_advantages = episode_data
        
            for frame in zip(episode_boards, episode_pieces, episode_actions, episode_probs, episode_returns, episode_advantages):
                board, pieces, action, probs, ret, advantage = frame
                self.replay_buffer.add_batch((board[None, ...], pieces[None, ...], action[None, ...], probs[None, ...], ret[None, ...], advantage[None, ...]))
            print(f'\rCurrent Episode: {i}', end='', flush=True)
        print('\rDone filling replay buffer', end='\n', flush=True)

    def train(self, gens, train_steps=100, training_actor=False, max_episode_steps=self.max_episode_steps):
        for gen in range(gens):
            # Run episode
            episode_data = self.player.run_episode(self.model, max_steps=max_episode_steps, greedy=False, renderer=self.renderer)
            episode_boards, episode_pieces, episode_actions, episode_probs, episode_rewards, episode_returns, _, episode_advantages = episode_data
        
            for frame in zip(episode_boards, episode_pieces, episode_actions, episode_probs, episode_returns, episode_advantages):
                board, pieces, action, probs, ret, advantage = frame
                self.replay_buffer.add_batch((board[None, ...], pieces[None, ...], action[None, ...], probs[None, ...], ret[None, ...], advantage[None, ...]))
        
            avg_reward = tf.reduce_mean(episode_rewards)
            
            print(f'\rCurrent Gen: {gen + 1}\t|\tReward: {avg_reward:1.1f}\t|', end='\n', flush=True)
        
            # Make dataset sampling from replay buffer
            dset = self.replay_buffer.as_dataset(
                sample_batch_size=128,
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)
            
            for i, ((board_batch, piece_batch, action_batch, old_probs, return_batch, advantage_batch), _) in enumerate(dset.take(train_steps)):
                losses = ppo_train_step(board_batch, piece_batch, action_batch, old_probs, return_batch, advantage_batch, training_actor)

            if training_actor:
                actor_loss, critic_loss = losses
                print(f'\rActor Loss: {actor_loss:1.2f}\t|\tCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)
                wandb.log({'actor_loss': actor_loss,
                           'critic_loss': critic_loss,
                           'reward': avg_reward})
            else:
                critic_loss = losses
                print(f'\rCritic Loss: {critic_loss:1.2f}\t|\t', end='', flush=True)
                wandb.log({'critic_loss': critic_loss,
                           'reward': avg_reward})

    