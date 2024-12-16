import tensorflow as tf
from TetrisEnv import TetrisEnv
from TetrisEnv import CustomScorer

class Player():
    def __init__(self, max_len):
        self.game = TetrisEnv(CustomScorer())
        self.reward_eps = tf.constant(0.01)
        self.hole_reward = tf.constant(0.1)
        self.bumpy_reward = tf.constant(0.02)
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

    def _get_heights(self, board):
        row_positions = tf.range(tf.shape(board)[0], 0, -1, dtype=tf.int32)[..., None]
        weighted_board = board * row_positions
        heights = tf.reduce_max(weighted_board, axis=0)
        return heights

    def _get_holes(self, board, heights):
        return tf.reduce_sum(heights - tf.reduce_sum(board, axis=0))

    def _get_bumpiness(self, heights):
        return tf.reduce_sum(tf.experimental.numpy.diff(heights) ** 2)
    
    def _get_supp_reward(self, board, last_holes, last_bumpiness):
        heights = self._get_heights(board)
        holes = self._get_holes(board, heights)
        bumpiness = self._get_bumpiness(heights)
        hole_reward = self.hole_reward if last_holes == holes else self.hole_reward * tf.cast(last_holes - holes, tf.float32)
        bumpy_reward = self.bumpy_reward if last_bumpiness == bumpiness else self.bumpy_reward * tf.cast(last_bumpiness - bumpiness, tf.float32)
        return holes, bumpiness, hole_reward, bumpy_reward
    
    def run_episode(self, agent, critic, max_steps=50, greedy=False, temperature=1.0, renderer=None):
        episode_boards = []
        episode_pieces = []
        episode_inputs = []
        episode_probs = []
        episode_values = []
        episode_rewards = []
    
        board, piece, _, terminated = self.game.reset()
        heights = self._get_heights(board)
        last_holes = self._get_holes(board, heights)
        last_bumpiness = self._get_bumpiness(heights)
        
        for t in range(max_steps):
            board_obs = tf.cast(board, tf.float32)[..., None]
            piece_obs = tf.cast(piece, tf.int32)
            inp_seq = tf.cast([[11]], tf.int32)
            key_chars = []
            
            agent_board_rep, _ = agent.process_board((board_obs[None, ...], piece_obs[None, ...]), training=False)
            critic_board_rep, _ = critic.process_board((board_obs[None, ...], piece_obs[None, ...]), training=False)
            
            for i in range(self.max_len):
                logits, _ = agent.process_keys((agent_board_rep, inp_seq), training=False)
                values, _ = critic.process_keys((critic_board_rep, inp_seq), training=False)
                
                if greedy:
                    key = tf.argmax(logits[:, -1:], axis=-1, output_type=tf.int32) # (1, 1)
                else:
                    key = tf.random.categorical(logits[:, -1] / temperature, num_samples=1, dtype=tf.int32) # (1, 1)
                
                inp_seq = tf.concat([inp_seq, key], axis=-1)
                key = tf.squeeze(key).numpy()
                key_chars.append(self.key_dict[key])
                
                if key == 8:
                    break

            key_chars[-1] = 'H'
            board, piece, attack, terminated = self.game.step(key_chars)
            last_holes, last_bumpiness, hole_reward, bumpy_reward = self._get_supp_reward(board, last_holes, last_bumpiness)
            scaled_attack = (attack ** 2) / 8.0
            
            episode_boards.append(board_obs)
            episode_pieces.append(piece_obs)
            episode_inputs.append(self._pad(inp_seq[0], self.max_len+1)) # (max_len+1,)
            chosen_probs = tf.gather(tf.nn.log_softmax(logits, axis=-1),
                                     inp_seq[:, 1:],
                                     batch_dims=2) # (1, len)
            episode_probs.append(self._pad(chosen_probs[0], self.max_len)[..., None]) # (max_len, 1)
            episode_values.append(values[0, -1]) # (1,)
            episode_rewards.append((hole_reward + bumpy_reward + scaled_attack + self.reward_eps)[None]) # (1,)
            
            if renderer:
                fig, img = renderer
                img.set_data(board)
                fig.canvas.draw()
                fig.canvas.flush_events()
    
            if terminated:
                episode_rewards[-1] = tf.constant([-5.0])
                break
        
        episode_boards = tf.stack(episode_boards, axis=0)
        episode_pieces = tf.stack(episode_pieces, axis=0)
        episode_inputs = tf.stack(episode_inputs, axis=0)
        episode_probs = tf.stack(episode_probs, axis=0)
        episode_values = tf.stack(episode_values, axis=0)
        episode_rewards = tf.stack(episode_rewards, axis=0)
        
        return episode_boards, episode_pieces, episode_inputs, episode_probs, episode_values, episode_rewards