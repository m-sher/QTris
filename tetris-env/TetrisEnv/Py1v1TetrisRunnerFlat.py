from TetrisEnv.PyTetris1v1Env import PyTetris1v1Env
import tensorflow as tf
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
from typing import Optional, Tuple, Any


class Py1v1TetrisRunnerFlat:
    """1v1 trajectory collector using flat (non-autoregressive) policy models."""

    def __init__(
        self,
        queue_size: int,
        max_holes: Optional[int],
        max_height: int,
        max_steps: int,
        max_len: int,
        num_steps: int,
        num_envs: int,
        p_model: Any,
        opp_model: Any,
        v_model: Any,
        temperature: float = 1.0,
        seed: int = 123,
        num_sequences: int = 160,
        num_row_tiers: int = 2,
        b2b_gap_coef: float = 0.0,
    ) -> None:
        self._queue_size = queue_size
        self._max_len = max_len
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._num_sequences = num_sequences
        self._num_row_tiers = num_row_tiers

        self.p_model = p_model
        self.opp_model = opp_model
        self.v_model = v_model
        self._temperature = temperature

        constructors = [
            lambda idx=i: PyTetris1v1Env(
                queue_size=queue_size,
                max_holes=max_holes,
                max_height=max_height,
                max_steps=max_steps,
                max_len=max_len,
                pathfinding=True,
                seed=seed + idx if seed is not None else None,
                idx=idx,
                num_row_tiers=num_row_tiers,
                b2b_gap_coef=b2b_gap_coef,
            )
            for i in range(num_envs)
        ]
        ppy_env = ParallelPyEnvironment(
            constructors, start_serially=True, blocking=False
        )
        self.env = TFPyEnvironment(ppy_env)

    def collect_trajectory(
        self, render: bool = False, progress: bool = False
    ) -> Tuple[tf.Tensor, ...]:
        all_boards = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 24, 10, 1),
        )
        all_pieces = tf.TensorArray(
            dtype=tf.int64, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, self._queue_size + 2),
        )
        all_b2b_combo_garbage = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 3),
        )
        all_log_probs = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_valid_sequences = tf.TensorArray(
            dtype=tf.int64, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, self._num_sequences, self._max_len),
        )
        all_action_indices = tf.TensorArray(
            dtype=tf.int64, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_values = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 1),
        )
        all_attacks = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_net_attacks = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_clears = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_attack_reward = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_total_reward = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_dones = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 1),
        )
        all_garbage_pushed = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 1),
        )
        all_wins = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_losses = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        # Opponent state for asymmetric value model training
        all_opp_boards = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 24, 10, 1),
        )
        all_opp_pieces = tf.TensorArray(
            dtype=tf.int64, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, self._queue_size + 2),
        )
        all_opp_b2b_combo_garbage = tf.TensorArray(
            dtype=tf.float32, size=self._num_steps, dynamic_size=False,
            element_shape=(self._num_envs, 3),
        )

        # Initialize pygame
        if render and not pygame.get_init():
            pygame.init()
            screen = pygame.display.set_mode((250, 600))
            pygame.display.set_caption("Tetris 1v1 Flat")
        else:
            screen = pygame.display.get_surface()

        time_step = self.env.reset()

        step_iter = range(self._num_steps)
        if progress:
            from tqdm import tqdm
            step_iter = tqdm(step_iter, desc="Collecting", unit="step")

        for t in step_iter:
            # Player 1 observations
            board = time_step.observation["board"]
            pieces = time_step.observation["pieces"]
            b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
            valid_sequences = time_step.observation["sequences"]

            # Player 2 observations
            opp_board = time_step.observation["opp_board"]
            opp_pieces = time_step.observation["opp_pieces"]
            opp_b2b_combo_garbage = time_step.observation["opp_b2b_combo_garbage"]
            opp_sequences = time_step.observation["opp_sequences"]

            # Render
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                screen.fill((0, 0, 0))
                board_surf = pygame.Surface((10, 24))
                pygame.surfarray.blit_array(
                    board_surf, board[0, ..., 0].numpy().T * 255
                )
                board_surf = pygame.transform.scale(board_surf, (250, 600))
                screen.blit(board_surf, (0, 0))
                pygame.display.update()

            # --- Player 1 action (training policy, flat) ---
            p1_key_sequence, log_prob, action_index, _ = self.p_model.predict(
                (board, pieces, b2b_combo_garbage),
                valid_sequences=valid_sequences,
                temperature=self._temperature,
            )

            # --- Player 2 action (opponent policy, flat, no data collected) ---
            p2_key_sequence, _, _, _ = self.opp_model.predict(
                (opp_board, opp_pieces, opp_b2b_combo_garbage),
                valid_sequences=opp_sequences,
                temperature=self._temperature,
            )

            # --- Asymmetric value estimate (sees both boards) ---
            values = self.v_model.predict(
                (board, pieces, b2b_combo_garbage, opp_board, opp_pieces, opp_b2b_combo_garbage)
            )

            # --- Step env with combined actions ---
            combined_action = tf.concat([p1_key_sequence, p2_key_sequence], axis=-1)
            time_step = self.env.step(combined_action)

            reward = time_step.reward
            attack = reward["attack"]
            net_attack = reward["net_attack"]
            clear = reward["clear"]
            attack_reward = reward["attack_reward"]
            total_reward = reward["total_reward"]
            garbage_pushed = reward["garbage_pushed"][..., None]
            win = reward["win"]
            loss = reward["loss"]

            dones = tf.cast(time_step.is_last(), tf.float32)[..., None]

            all_boards = all_boards.write(t, board)
            all_pieces = all_pieces.write(t, pieces)
            all_b2b_combo_garbage = all_b2b_combo_garbage.write(t, b2b_combo_garbage)
            all_log_probs = all_log_probs.write(t, log_prob)
            all_valid_sequences = all_valid_sequences.write(t, valid_sequences)
            all_action_indices = all_action_indices.write(t, action_index)
            all_values = all_values.write(t, values)

            all_attacks = all_attacks.write(t, attack)
            all_net_attacks = all_net_attacks.write(t, net_attack)
            all_clears = all_clears.write(t, clear)
            all_attack_reward = all_attack_reward.write(t, attack_reward)
            all_total_reward = all_total_reward.write(t, total_reward)

            all_dones = all_dones.write(t, dones)
            all_garbage_pushed = all_garbage_pushed.write(t, garbage_pushed)
            all_wins = all_wins.write(t, win)
            all_losses = all_losses.write(t, loss)

            all_opp_boards = all_opp_boards.write(t, opp_board)
            all_opp_pieces = all_opp_pieces.write(t, opp_pieces)
            all_opp_b2b_combo_garbage = all_opp_b2b_combo_garbage.write(t, opp_b2b_combo_garbage)

        # Bootstrap
        board = time_step.observation["board"]
        pieces = time_step.observation["pieces"]
        b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
        opp_board = time_step.observation["opp_board"]
        opp_pieces = time_step.observation["opp_pieces"]
        opp_b2b_combo_garbage = time_step.observation["opp_b2b_combo_garbage"]
        all_last_values = self.v_model.predict(
            (board, pieces, b2b_combo_garbage, opp_board, opp_pieces, opp_b2b_combo_garbage)
        )

        all_boards = all_boards.stack()
        all_pieces = all_pieces.stack()
        all_b2b_combo_garbage = all_b2b_combo_garbage.stack()
        all_log_probs = all_log_probs.stack()
        all_valid_sequences = all_valid_sequences.stack()
        all_action_indices = all_action_indices.stack()
        all_values = all_values.stack()
        all_attacks = all_attacks.stack()
        all_net_attacks = all_net_attacks.stack()
        all_clears = all_clears.stack()
        all_attack_reward = all_attack_reward.stack()
        all_total_reward = all_total_reward.stack()
        all_dones = all_dones.stack()
        all_garbage_pushed = all_garbage_pushed.stack()
        all_wins = all_wins.stack()
        all_losses = all_losses.stack()
        all_opp_boards = all_opp_boards.stack()
        all_opp_pieces = all_opp_pieces.stack()
        all_opp_b2b_combo_garbage = all_opp_b2b_combo_garbage.stack()

        return (
            all_boards,
            all_pieces,
            all_b2b_combo_garbage,
            all_log_probs,
            all_valid_sequences,
            all_action_indices,
            all_values,
            all_last_values,
            all_attacks,
            all_net_attacks,
            all_clears,
            all_attack_reward,
            all_total_reward,
            all_dones,
            all_garbage_pushed,
            all_wins,
            all_losses,
            # Opponent state for value model training
            all_opp_boards,
            all_opp_pieces,
            all_opp_b2b_combo_garbage,
        )
