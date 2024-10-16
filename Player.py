import tensorflow as tf
from TetrisEnv import TetrisEnv
from TetrisEnv import CustomScorer

class Player():
    def __init__(self):
        self.game = TetrisEnv(CustomScorer())
        self.eps = 1e-10
    
    def run_episode(self, agent, max_steps=50, greedy=False, renderer=None):
        episode_boards = []
        episode_pieces = []
        episode_actions = []
        episode_probs = []
        episode_values = []
        episode_rewards = []
    
        board, piece, _, terminated = self.game.reset()
        
        for t in range(max_steps):
            board_obs = tf.cast(board, tf.float32)[..., None]
            piece_obs = tf.cast(piece, tf.int32)
            inp_seq = tf.cast([[11]], tf.int32)
            key_chars = []
            board_rep, _ = agent.process_board((board_obs[None, ...], piece_obs[None, ...]), training=False)
            for _ in range(max_len-1):
                logits, _ = agent.process_keys((board_rep, inp_seq), training=False)
                values, _ = agent.process_vals(board_rep, training=False)
                
                if greedy or tf.random.uniform(()) > 0.1:
                    key = tf.argmax(logits[:, -1:], axis=-1, output_type=tf.int32) # (1, 1)
                else:
                    key = tf.random.categorical(logits[:, -1], num_samples=1, dtype=tf.int32) # (1, 1)
    
                inp_seq = tf.concat([inp_seq, key], axis=-1)
    
                key = tf.squeeze(key).numpy()
                key_chars.append(key_dict[key])
                if key == 8:
                    break
    
            if key_chars[-1] != 'H':
                key_chars[-1] = 'H'
                inp_seq = tf.concat([inp_seq[:, :-1], [[8]]], axis=-1)
    
            # 1, 1
            chosen_prob = tf.gather(tf.nn.log_softmax(logits, axis=-1),
                                    inp_seq[..., 1:],
                                    batch_dims=2)
            
            board, piece, reward, terminated = self.game.step(key_chars)
    
            if renderer:
                fig, img = renderer
                img.set_data(board)
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            episode_boards.append(board_obs)
            episode_pieces.append(piece_obs)
            episode_actions.append(pad(inp_seq[0], max_len))
            episode_probs.append(pad(chosen_prob[0], max_len-1))
            episode_values.append(values[0])
            episode_rewards.append(reward + self.eps)
    
            if terminated:
                break
    
        if not terminated:
            episode_rewards[-1] = episode_values[-1][0]
        
        episode_boards = tf.stack(episode_boards, axis=0)
        episode_pieces = tf.stack(episode_pieces, axis=0)
        episode_actions = tf.stack(episode_actions, axis=0)
        episode_probs = tf.stack(episode_probs, axis=0)[..., None]
        episode_rewards = tf.stack(episode_rewards, axis=0)
        episode_returns = self.get_expected_return(episode_rewards, gamma)[..., None]
        episode_values = tf.stack(episode_values, axis=0)
        episode_advantages = episode_returns - episode_values
        
        return (episode_boards, episode_pieces, episode_actions, episode_probs, 
                episode_rewards, episode_returns, episode_values, episode_advantages)

    @tf.function
    def get_expected_return(rewards, gamma):
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            returns = returns.write(i, discounted_sum)
            
        returns = returns.stack()[::-1]
    
        return returns