import tensorflow as tf
from TetrisEnv import TetrisEnv
from TetrisEnv import CustomScorer

class Player():
    def __init__(self, max_len):
        self.game = TetrisEnv(CustomScorer())
        self.eps = 1e-10
        self.max_len = max_len
        
        self.key_dict = {
            0: 'N',
            1: 'l',
            2: 'r',
            3: 'L',
            4: 'R',
            5: 's',
            6: 'a',
            7: 'c',
            8: 'H',
            9: 'h',
            10: '1',
            11: 'S',
        }

    def _pad(self, item, length, pad_value=0):
        num_valid = tf.shape(item)[0]
        if num_valid > length:
            padded = item[:length]
        else:
            padded = tf.concat([item, tf.zeros((length - num_valid), dtype=item.dtype) + pad_value], axis=0)
        return padded
    
    def run_episode(self, agent, max_steps=50, greedy=False, renderer=None):
        episode_boards = []
        episode_pieces = []
        episode_inputs = []
        episode_actions = []
        episode_valid = []
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
            
            for i in range(self.max_len):
                logits, _ = agent.process_keys((board_rep, inp_seq), training=False)
                values, _ = agent.process_vals((board_rep, inp_seq), training=False)
                
                if greedy or tf.random.uniform(()) > 0.1:
                    key = tf.argmax(logits[:, -1:], axis=-1, output_type=tf.int32) # (1, 1)
                else:
                    key = tf.random.categorical(logits[:, -1], num_samples=1, dtype=tf.int32) # (1, 1)

                episode_boards.append(board_obs)
                episode_pieces.append(piece_obs)
                episode_inputs.append(self._pad(inp_seq[0], self.max_len))
                episode_actions.append(key[0, 0])
                episode_valid.append(i)
                episode_probs.append(tf.nn.log_softmax(logits, axis=-1)[0, -1, key[0, 0]])
                episode_values.append(values[0, -1])
                episode_rewards.append(0.0)
                
                inp_seq = tf.concat([inp_seq, key], axis=-1)
                key = tf.squeeze(key).numpy()
                key_chars.append(self.key_dict[key])
                
                if key == 8:
                    break

            key_chars[-1] = 'H'
            board, piece, reward, terminated = self.game.step(key_chars)
            episode_rewards[-1] = reward / 4.0 + 0.01
            
            if renderer:
                fig, img = renderer
                img.set_data(board)
                fig.canvas.draw()
                fig.canvas.flush_events()
    
            if terminated:
                break
        
        episode_boards = tf.stack(episode_boards, axis=0)
        episode_pieces = tf.stack(episode_pieces, axis=0)
        episode_inputs = tf.stack(episode_inputs, axis=0)
        episode_actions = tf.stack(episode_actions, axis=0)
        episode_valid = tf.stack(episode_valid, axis=0)
        episode_probs = tf.stack(episode_probs, axis=0)[..., None]
        episode_values = tf.stack(episode_values, axis=0)
        episode_rewards = tf.stack(episode_rewards, axis=0)[..., None]
        
        return episode_boards, episode_pieces, episode_inputs, episode_actions, episode_valid, episode_probs, episode_values, episode_rewards