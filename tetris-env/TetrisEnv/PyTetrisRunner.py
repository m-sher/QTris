from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Moves import Convert
import tensorflow as tf
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
from typing import Optional, Tuple, Any


class PyTetrisRunner:
    def __init__(
        self,
        queue_size: int,
        max_holes: Optional[int],
        max_height: int,
        max_steps: int,
        pathfinding: bool,
        max_len: int,
        key_dim: int,
        num_steps: int,
        num_envs: int,
        garbage_chance_min: float,
        garbage_chance_max: float,
        garbage_rows_min: int,
        garbage_rows_max: int,
        p_model: Any,
        v_model: Any,
        temperature: float = 1.0,
        seed: int = 123,
        num_sequences: int = 160,
        num_row_tiers: int = 1,
    ) -> None:
        self._queue_size = queue_size
        self._max_len = max_len
        self._key_dim = key_dim
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._pathfinding = pathfinding
        self._num_sequences = num_sequences
        self._num_row_tiers = num_row_tiers

        self.p_model = p_model
        self.v_model = v_model
        self._temperature = temperature

        garbage_chances = [
            garbage_chance_min
            + (garbage_chance_max - garbage_chance_min) * i / (num_envs - 1)
            for i in range(num_envs)
        ]

        constructors = [
            lambda idx=i: PyTetrisEnv(
                queue_size=queue_size,
                max_holes=max_holes,
                max_height=max_height,
                max_steps=max_steps,
                max_len=max_len,
                pathfinding=pathfinding,
                seed=seed,
                idx=idx,
                garbage_chance=garbage_chances[idx],
                garbage_min=garbage_rows_min,
                garbage_max=garbage_rows_max,
                num_row_tiers=num_row_tiers,
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
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 24, 10, 1),
        )
        all_pieces = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._queue_size + 2),
        )
        all_b2b_combo_garbage = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 3),
        )
        all_actions = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._max_len),
        )
        all_log_probs = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._max_len),
        )
        all_masks = tf.TensorArray(
            dtype=tf.bool,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._max_len, self._key_dim),
        )
        all_valid_sequences = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._num_sequences, self._max_len),
        )
        all_action_indices = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_values = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 1),
        )
        all_attacks = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_clears = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_attack_reward = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_total_reward = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_dones = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 1),
        )
        all_garbage_pushed = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 1),
        )

        # Initialize pygame
        if render and not pygame.get_init():
            pygame.init()
            screen = pygame.display.set_mode((250, 600))
            pygame.display.set_caption("Tetris")
        else:
            screen = pygame.display.get_surface()

        time_step = self.env.reset()

        step_iter = range(self._num_steps)
        if progress:
            from tqdm import tqdm
            step_iter = tqdm(step_iter, desc="Collecting", unit="step")

        for t in step_iter:
            board = time_step.observation["board"]
            pieces = time_step.observation["pieces"]
            b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
            valid_sequences = time_step.observation["sequences"]

            # Render the frame
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

            if self._pathfinding:
                key_sequence, log_probs, masks, _ = self.p_model.predict(
                    (board, pieces, b2b_combo_garbage), valid_sequences=valid_sequences, temperature=self._temperature
                )
            else:
                key_sequence, log_probs, masks, _ = self.p_model.predict(
                    (board, pieces, b2b_combo_garbage), valid_sequences=Convert.tf_to_sequences[None, ...], temperature=self._temperature
                )

            matches = tf.reduce_all(
                tf.equal(key_sequence[:, None, :], valid_sequences), axis=-1
            )
            action_index = tf.argmax(tf.cast(matches, tf.int64), axis=-1)

            values = self.v_model.predict((board, pieces, b2b_combo_garbage))

            time_step = self.env.step(key_sequence)

            reward = time_step.reward
            attack = reward["attack"]
            clear = reward["clear"]
            attack_reward = reward["attack_reward"]
            total_reward = reward["total_reward"]
            garbage_pushed = reward["garbage_pushed"][..., None]

            dones = tf.cast(time_step.is_last(), tf.float32)[..., None]

            all_boards = all_boards.write(t, board)
            all_pieces = all_pieces.write(t, pieces)
            all_b2b_combo_garbage = all_b2b_combo_garbage.write(t, b2b_combo_garbage)
            all_actions = all_actions.write(t, key_sequence)
            all_log_probs = all_log_probs.write(t, log_probs)
            all_masks = all_masks.write(t, masks)
            all_valid_sequences = all_valid_sequences.write(t, valid_sequences)
            all_action_indices = all_action_indices.write(t, action_index)
            all_values = all_values.write(t, values)

            # Store the penalties and rewards
            all_attacks = all_attacks.write(t, attack)
            all_clears = all_clears.write(t, clear)
            all_attack_reward = all_attack_reward.write(t, attack_reward)
            all_total_reward = all_total_reward.write(t, total_reward)

            all_dones = all_dones.write(t, dones)
            all_garbage_pushed = all_garbage_pushed.write(t, garbage_pushed)

        # bootstrap
        board = time_step.observation["board"]
        pieces = time_step.observation["pieces"]
        b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
        all_last_values = self.v_model.predict((board, pieces, b2b_combo_garbage))

        all_boards = all_boards.stack()
        all_pieces = all_pieces.stack()
        all_b2b_combo_garbage = all_b2b_combo_garbage.stack()
        all_actions = all_actions.stack()
        all_log_probs = all_log_probs.stack()
        all_masks = all_masks.stack()
        all_valid_sequences = all_valid_sequences.stack()
        all_action_indices = all_action_indices.stack()
        all_values = all_values.stack()
        all_attacks = all_attacks.stack()
        all_clears = all_clears.stack()
        all_attack_reward = all_attack_reward.stack()
        all_total_reward = all_total_reward.stack()
        all_dones = all_dones.stack()
        all_garbage_pushed = all_garbage_pushed.stack()

        return (
            all_boards,
            all_pieces,
            all_b2b_combo_garbage,
            all_actions,
            all_log_probs,
            all_masks,
            all_valid_sequences,
            all_action_indices,
            all_values,
            all_last_values,
            all_attacks,
            all_clears,
            all_attack_reward,
            all_total_reward,
            all_dones,
            all_garbage_pushed,
        )
