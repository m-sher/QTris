import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from TetrisEnv import TetrisEnv
from TetrisEnv import CustomScorer
import pygame

class Player():
    def __init__(self, max_len, num_players, players_to_render, max_holes, reward_eps=0.01):
        self._games = [TetrisEnv(CustomScorer()) for _ in range(num_players)]
        self._num_players = num_players
        self._players_to_render = players_to_render
        self._max_holes = max_holes
        self._reward_eps = reward_eps
        self._hole_reward = 0.1
        self._bumpy_reward = 0.05
        self._max_len = max_len
        
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
        self.screen = pygame.display.set_mode((250*players_to_render, 700))
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
        hole_reward = self._hole_reward if last_holes >= holes else 0.0
        bumpy_reward = self._bumpy_reward if last_bumpiness >= bumpiness else 0.0
        return holes, bumpiness, hole_reward, bumpy_reward
    
    def _single_start(self, game):
        episode_boards = []
        episode_pieces = []
        episode_inputs = []
        episode_actions = []
        episode_probs = []
        episode_values = []
        episode_rewards = []

        episode_data = episode_boards, episode_pieces, episode_inputs, episode_actions, episode_probs, episode_values, episode_rewards
        
        board, piece, _, terminated = game.reset()
        heights = self._get_heights(board)
        last_holes = self._get_holes(board, heights)
        last_bumpiness = self._get_bumpiness(heights)
        
        return episode_data, board, piece, terminated, last_holes, last_bumpiness
    
    def _parallel_start(self):
        with ThreadPoolExecutor() as executor:
            results = executor.map(self._single_start, self._games)

        return list(results)

    def _single_step(self, living, game, key_chars, last_holes, last_bumpiness):
        if living:
            key_chars[-1] = 'H'
            board, piece, attack, terminated = game.step(key_chars)
            last_holes, last_bumpiness, hole_reward, bumpy_reward = self._get_supp_reward(board, last_holes, last_bumpiness)
            scaled_attack = (attack ** 2) / 8.0
            env_rewards = np.array([hole_reward + bumpy_reward + scaled_attack + self._reward_eps], dtype=np.float32)
        else:
            board, piece, attack, terminated = game.current_time_step()
            env_rewards = None

        return board, piece, terminated, last_holes, last_bumpiness, env_rewards
    
    def _parallel_step(self, living_players, all_key_chars, all_last_holes, all_last_bumpiness):
        with ThreadPoolExecutor() as executor:
            futures = []
            for player, (game, key_chars, last_holes, last_bumpiness) in enumerate(zip(self._games,
                                                                                       all_key_chars,
                                                                                       all_last_holes,
                                                                                       all_last_bumpiness)):

                futures.append(executor.submit(self._single_step, player in living_players, game,
                                               key_chars, last_holes, last_bumpiness))
                
            results = [future.result() for future in futures]
        return results

    def _single_log(self, episode_data, player, processing_players, log_data, key_chars):
        (episode_boards, episode_pieces, episode_inputs,
         episode_actions, episode_probs, episode_values, episode_rewards) = episode_data
        
        board, piece, inp, key, prob, value = log_data
        
        episode_boards.append(board)
        episode_pieces.append(piece)
        episode_inputs.append(self._pad(inp, self._max_len))
        episode_actions.append(key)
        episode_probs.append(prob)
        episode_values.append(value)
        episode_rewards.append(np.array([-self._reward_eps], dtype=np.float32))
        
        key_chars.append(self.key_dict[key])
        
        if key == 8:
            processing_players.remove(player)

    def _parallel_log(self, all_episode_data, processing_players, all_log_data, all_key_chars):
        (all_episode_boards, all_episode_pieces, all_episode_inputs,
         all_episode_actions, all_episode_probs, all_episode_values, all_episode_rewards) = all_episode_data
        
        all_board, all_piece, all_inputs, all_keys, all_probs, all_values = all_log_data
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for player in processing_players[:]:
                episode_data = [all_episode_boards[player],
                                all_episode_pieces[player],
                                all_episode_inputs[player],
                                all_episode_actions[player],
                                all_episode_probs[player],
                                all_episode_values[player],
                                all_episode_rewards[player]]
                log_data = [all_board[player],
                            all_piece[player],
                            all_inputs[player],
                            all_keys[player],
                            all_probs[player],
                            all_values[player]]
                futures.append(executor.submit(self._single_log, episode_data, player,
                                               processing_players, log_data, all_key_chars[player]))
            
            

    def run_episode(self, actor, critic, max_steps=50, greedy=False, temperature=1.0):

        living_players = list(range(self._num_players))
        
        all_episode_data, all_board, all_piece, all_terminated, all_last_holes, all_last_bumpiness = zip(*self._parallel_start())

        (all_episode_boards, all_episode_pieces, all_episode_inputs,
         all_episode_actions, all_episode_probs, all_episode_values, all_episode_rewards) = list(map(list, zip(*all_episode_data)))
        
        all_episode_data = [all_episode_boards,
                            all_episode_pieces,
                            all_episode_inputs,
                            all_episode_actions,
                            all_episode_probs,
                            all_episode_values,
                            all_episode_rewards]
        
        board_obs = tf.cast(tf.stack(all_board, axis=0), tf.float32)[..., None]
        piece_obs = tf.cast(tf.stack(all_piece, axis=0), tf.int32)

        for t in range(max_steps):
            
            inp_seq = tf.cast([[11] for _ in range(self._num_players)], tf.int32)
            all_key_chars = [[] for _ in range(self._num_players)]
            
            actor_board_rep, _ = actor.process_obs((board_obs, piece_obs), training=False)
            critic_board_rep, _ = critic.process_obs((board_obs, piece_obs), training=False)

            processing_players = [player for player in living_players]

            for i in range(self._max_len):
                
                logits, _ = actor.process_keys((actor_board_rep, inp_seq), training=False)
                values, _ = critic.process_keys((critic_board_rep, inp_seq), training=False)
                
                if greedy:
                    keys = tf.argmax(logits[:, -1:], axis=-1, output_type=tf.int32) # (num_players, 1)
                else:
                    keys = tf.random.categorical(logits[:, -1] / temperature, num_samples=1, dtype=tf.int32) # (num_players, 1)

                probs = tf.nn.log_softmax(logits[:, -1], axis=-1).numpy()
                all_log_data = [all_board,
                                all_piece,
                                inp_seq.numpy(),
                                keys[:, 0].numpy(),
                                probs,
                                values[:, -1].numpy()]
                
                self._parallel_log(all_episode_data, processing_players, all_log_data, all_key_chars)

                inp_seq = tf.concat([inp_seq, keys], axis=-1)
                
                if len(processing_players) == 0:
                    break

            step_out = self._parallel_step(living_players, all_key_chars, all_last_holes, all_last_bumpiness)
            all_board, all_piece, all_terminated, all_last_holes, all_last_bumpiness, env_rewards = list(map(list, zip(*step_out)))

            board_obs = tf.cast(tf.stack(all_board, axis=0), tf.float32)[..., None]
            piece_obs = tf.cast(tf.stack(all_piece, axis=0), tf.int32)

            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.screen.fill((0, 0, 0))
            
            for i, board in zip(range(self._players_to_render), all_board):
                board_surface = pygame.Surface((10, 28))
                pygame.surfarray.blit_array(board_surface, board.T * 255)
                board_surface = pygame.transform.scale(board_surface, (250, 700))
                update_rect = self.screen.blit(board_surface, (250 * i, 0))
                pygame.display.update(update_rect)
    
            for player in living_players[:]:
                if all_terminated[player] or all_last_holes[player] > self._max_holes:
                    all_episode_rewards[player][-1] = np.array([-1.0], dtype=np.float32)
                    living_players.remove(player)
                else:
                    all_episode_rewards[player][-1] = env_rewards[player]
            
            if len(living_players) == 0:
                break
        
        for player in range(self._num_players):
            all_episode_boards[player] = tf.stack(all_episode_boards[player])
            all_episode_pieces[player] = tf.stack(all_episode_pieces[player])
            all_episode_inputs[player] = tf.stack(all_episode_inputs[player])
            all_episode_actions[player] = tf.stack(all_episode_actions[player])
            all_episode_probs[player] = tf.stack(all_episode_probs[player])
            all_episode_values[player] = tf.stack(all_episode_values[player])
            all_episode_rewards[player] = tf.stack(all_episode_rewards[player])
        
        return all_episode_boards, all_episode_pieces, all_episode_inputs, all_episode_actions, all_episode_probs, all_episode_values, all_episode_rewards