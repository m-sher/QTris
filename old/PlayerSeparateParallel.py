import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from TetrisEnv import TetrisEnv
from TetrisEnv import CustomScorer
import pygame

class Player():
    def __init__(self, ind_to_str, num_players, players_to_render, max_holes, max_height, max_diff):
        self._ind_to_str = ind_to_str
        self._games = [TetrisEnv(CustomScorer()) for _ in range(num_players)]
        self._num_players = num_players
        self._players_to_render = players_to_render
        self._max_holes = max_holes
        self._max_height = max_height
        self._max_diff = max_diff
        self._clear_reward = 0.5
        self._hole_reward = 0.01
        self._bumpy_reward = 0.01
        self._height_reward = 0.01
        self._diff_reward = 0.01
        self._step_reward = 0.1
        
        pygame.init()
        self.screen = pygame.display.set_mode((250*players_to_render, 700))
        self.clock = pygame.time.Clock()
        self._render_interval = 1000 / 30

    def _get_heights(self, board):
        row_positions = np.arange(board.shape[0], 0, -1, dtype=np.int32)[..., None]
        weighted_board = board * row_positions
        heights = np.max(weighted_board, axis=0)
        return heights

    def _get_holes(self, board, heights):
        return np.sum(heights - np.sum(board, axis=0))

    def _get_bumpiness(self, heights):
        return np.sum(abs(np.diff(heights)))
    
    def _get_diff(self, heights):
        return np.max(heights) - np.min(heights)
    
    def _get_supp_reward(self, board, last_board, last_heights, last_holes, last_bumpiness, last_diff, terminated):
        heights = self._get_heights(board)
        holes = self._get_holes(board, heights)
        bumpiness = self._get_bumpiness(heights)
        diff = self._get_diff(heights)
        lines_cleared = (np.sum(last_board) + 4 - np.sum(board)) / 10
        clear_reward = self._clear_reward * lines_cleared
        hole_reward = self._hole_reward * (last_holes - holes + 1)
        bumpy_reward = self._bumpy_reward * (last_bumpiness - bumpiness + 1)
        height_reward = self._height_reward * (np.max(last_heights) - np.max(heights) + 1)
        diff_reward = self._diff_reward * (last_diff - diff + 1)
        return heights, holes, bumpiness, diff, clear_reward, hole_reward, bumpy_reward, height_reward, diff_reward
    
    def _single_start(self, game):
        episode_boards = []
        episode_pieces = []
        episode_actions = []
        episode_probs = []
        episode_values = []
        episode_rewards = []
        episode_dones = []

        episode_data = (episode_boards, episode_pieces, episode_actions,
                        episode_probs, episode_values, episode_rewards, episode_dones)
        
        board, piece, _, terminated = game.reset()
        last_heights = self._get_heights(board)
        last_holes = self._get_holes(board, last_heights)
        last_bumpiness = self._get_bumpiness(last_heights)
        last_diff = self._get_diff(last_heights)
        
        return episode_data, board, piece, terminated, last_heights, last_holes, last_bumpiness, last_diff
    
    def _parallel_start(self):
        with ThreadPoolExecutor() as executor:
            results = executor.map(self._single_start, self._games)

        return list(results)

    def _single_step(self, step_data):
        (game, last_board, piece, action, log_probs, value,
         last_heights, last_holes, last_bumpiness, last_diff,
         episode_boards, episode_pieces, episode_actions,
         episode_probs, episode_values, episode_rewards, episode_dones) = step_data
        
        key_chars = self._ind_to_str[tuple(action)]
        episode_boards.append(last_board)
        episode_pieces.append(piece)
        episode_actions.append(action)
        episode_probs.append(log_probs)
        episode_values.append(value)
        
        board, piece, attack, terminated = game.step(key_chars)
        last_heights, last_holes, last_bumpiness, last_diff, clear_reward, hole_reward, bumpy_reward, height_reward, diff_reward = self._get_supp_reward(board, last_board, last_heights, last_holes, last_bumpiness, last_diff, terminated)
        
        if (terminated or
            last_holes > self._max_holes or
            np.max(last_heights) > self._max_height or
            last_diff > self._max_diff):
            
            board, piece, _, _ = game.reset()
            last_heights = self._get_heights(board)
            last_holes = self._get_holes(board, last_heights)
            last_bumpiness = self._get_bumpiness(last_heights)
            last_diff = self._get_diff(last_heights)
            done = 1.0
            env_rewards = -1.0
        else:
            done = 0.0
            env_rewards = hole_reward + bumpy_reward + height_reward + diff_reward + attack + self._step_reward
        
        episode_dones.append(done)
        
        episode_rewards.append(np.array(env_rewards, np.float32))

        return board, piece, terminated, last_heights, last_holes, last_bumpiness, last_diff
    
    def _parallel_step(self, all_step_data):
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for player, step_data in enumerate(zip(self._games, *all_step_data)):

                futures.append(executor.submit(self._single_step, step_data))
                
            results = [future.result() for future in futures]
        return results

    def run_episode(self, model, max_steps=50, greedy=False, temperature=1.0):
        
        all_episode_data, all_board, all_piece, all_terminated, all_last_heights, all_last_holes, all_last_bumpiness, all_last_diff = zip(*self._parallel_start())

        (all_episode_boards, all_episode_pieces, all_episode_actions,
         all_episode_probs, all_episode_values, all_episode_rewards, all_episode_dones) = list(map(list, zip(*all_episode_data)))
        
        board_obs = tf.cast(tf.stack(all_board, axis=0), tf.float32)[..., None]
        piece_obs = tf.cast(tf.stack(all_piece, axis=0), tf.int32)

        last_ticks = pygame.time.get_ticks()

        for t in range(max_steps):

            all_actions, all_logits, all_values = model.predict((board_obs, piece_obs), greedy=greedy, temperature=temperature, training=False)

            distributions = tfp.distributions.Categorical(logits=all_logits / temperature)
            log_probs = tf.reduce_sum(distributions.log_prob(all_actions), axis=-1)

            all_step_data = [all_board, all_piece, all_actions.numpy(), log_probs.numpy(),
                             all_values.numpy(), all_last_heights, all_last_holes, all_last_bumpiness,
                             all_last_diff, all_episode_boards, all_episode_pieces, all_episode_actions,
                             all_episode_probs, all_episode_values, all_episode_rewards, all_episode_dones]

            step_out = self._parallel_step(all_step_data)
            
            all_board, all_piece, all_terminated, all_last_heights, all_last_holes, all_last_bumpiness, all_last_diff = list(map(list, zip(*step_out)))

            board_obs = tf.cast(tf.stack(all_board, axis=0), tf.float32)[..., None]
            piece_obs = tf.cast(tf.stack(all_piece, axis=0), tf.int32)

            self.clock.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
            cur_ticks = pygame.time.get_ticks()
            if cur_ticks - last_ticks >= self._render_interval:
                self.screen.fill((0, 0, 0))
                
                for i, board in zip(range(self._players_to_render), all_board):
                    board_surface = pygame.Surface((10, 28))
                    pygame.surfarray.blit_array(board_surface, board.T * 255)
                    board_surface = pygame.transform.scale(board_surface, (250, 700))
                    update_rect = self.screen.blit(board_surface, (250 * i, 0))
                    pygame.display.update(update_rect)
                    
                last_ticks = cur_ticks
        
        for player in range(self._num_players):
            all_episode_boards[player] = tf.stack(all_episode_boards[player])
            all_episode_pieces[player] = tf.stack(all_episode_pieces[player])
            all_episode_actions[player] = tf.stack(all_episode_actions[player])
            all_episode_probs[player] = tf.stack(all_episode_probs[player])
            all_episode_values[player] = tf.stack(all_episode_values[player])
            all_episode_rewards[player] = tf.stack(all_episode_rewards[player])
            all_episode_dones[player] = tf.stack(all_episode_dones[player])
        
        return (all_episode_boards, all_episode_pieces, all_episode_actions,
                all_episode_probs, all_episode_values, all_episode_rewards, all_episode_dones)