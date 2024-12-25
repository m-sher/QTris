import tensorflow as tf
import numpy as np
from TetrisEnv import TetrisEnv
from TetrisEnv import CustomScorer
import pygame


class Player():
    def __init__(self, max_len):
        self.game = TetrisEnv(CustomScorer())
        self.reward_eps = 0.01
        self.hole_reward = 0.1
        self.bumpy_reward = 0.02
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

        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

    def _pad(self, item, length, pad_value=0):
        num_valid = np.shape(item)[0]
        if num_valid > length:
            padded = item[:length]
        else:
            padded = np.concatenate([item, np.zeros((length - num_valid), dtype=item.dtype) + pad_value], axis=0)
        return padded

    def _get_heights(self, board):
        row_positions = np.arange(board.shape[0], 0, -1, dtype=np.int32)[..., None]
        weighted_board = board * row_positions
        heights = np.max(weighted_board, axis=0)
        return heights

    def _get_holes(self, board, heights):
        return np.sum(heights - np.sum(board, axis=0))

    def _get_bumpiness(self, heights):
        return np.sum(np.diff(heights) ** 2)
    
    def _get_supp_reward(self, board, last_holes, last_bumpiness):
        heights = self._get_heights(board)
        holes = self._get_holes(board, heights)
        bumpiness = self._get_bumpiness(heights)
        hole_reward = self.hole_reward if last_holes == holes else self.hole_reward * (last_holes - holes)
        bumpy_reward = self.bumpy_reward if last_bumpiness == bumpiness else self.bumpy_reward * (last_bumpiness - bumpiness)
        return holes, bumpiness, hole_reward, bumpy_reward
    
    def run_episode(self, agent, critic, max_steps=50, greedy=False, temperature=1.0):
        episode_boards = []
        episode_pieces = []
        episode_inputs = []
        episode_actions = []
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
                    key = tf.argmax(logits.numpy()[:, -1:], axis=-1, output_type=tf.int32) # (1, 1)
                else:
                    key = tf.random.categorical(logits.numpy()[:, -1] / temperature, num_samples=1, dtype=tf.int32) # (1, 1)

                episode_boards.append(board_obs)
                episode_pieces.append(piece_obs)
                episode_inputs.append(self._pad(inp_seq[0].numpy(), self.max_len))
                episode_actions.append(np.squeeze(key))
                episode_probs.append(np.array(tf.nn.log_softmax(logits[0, -1], axis=-1))[np.squeeze(key), None])
                episode_values.append(values[0, -1].numpy())
                episode_rewards.append(np.array([0.0]))
                
                inp_seq = tf.concat([inp_seq, key], axis=-1)
                key = np.squeeze(key)
                key_chars.append(self.key_dict[int(key)])
                
                if key == 8:
                    break

            key_chars[-1] = 'H'
            board, piece, attack, terminated = self.game.step(key_chars)
            last_holes, last_bumpiness, hole_reward, bumpy_reward = self._get_supp_reward(board, last_holes, last_bumpiness)
            scaled_attack = (attack ** 2) / 8.0

            episode_rewards[-1] = np.array(hole_reward + bumpy_reward + scaled_attack + self.reward_eps).astype(np.float32)[None]

            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            
            self.screen.fill((0, 0, 0))
            board_surface = pygame.Surface((10, 28))
            pygame.surfarray.blit_array(board_surface, board.T * 255)
            board_surface = pygame.transform.scale(board_surface, (214, 600))
            update_rect = self.screen.blit(board_surface, (191, 0))
            pygame.display.update(update_rect)
    
            if terminated:
                episode_rewards[-1] = np.array([-1.0])
                break
        
        episode_boards = np.stack(episode_boards, axis=0)
        episode_pieces = np.stack(episode_pieces, axis=0)
        episode_inputs = np.stack(episode_inputs, axis=0)
        episode_actions = np.stack(episode_actions, axis=0)
        episode_probs = np.stack(episode_probs, axis=0)
        episode_values = np.stack(episode_values, axis=0)
        episode_rewards = np.stack(episode_rewards, axis=0)
        
        return episode_boards, episode_pieces, episode_inputs, episode_actions, episode_probs, episode_values, episode_rewards