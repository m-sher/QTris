from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Moves import Convert, Keys
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
        max_len: int,
        num_steps: int,
        num_envs: int,
        garbage_chance_min: float,
        garbage_chance_max: float,
        garbage_rows_min: int,
        garbage_rows_max: int,
        p_model: Any,
        v_model: Any,
        seed: int | None = None,
    ) -> None:
        self._queue_size = queue_size
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._max_len = max_len
        self.p_model = p_model
        self.v_model = v_model

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
                seed=seed,
                idx=idx,
                garbage_chance=garbage_chances[idx],
                garbage_min=garbage_rows_min,
                garbage_max=garbage_rows_max,
            )
            for i in range(num_envs)
        ]
        ppy_env = ParallelPyEnvironment(
            constructors, start_serially=True, blocking=False
        )
        self.env = TFPyEnvironment(ppy_env)

    def collect_trajectory(
        self, render: bool = False
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
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
        all_b2b_combo = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 2),
        )
        all_actions = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_masks = tf.TensorArray(
            dtype=tf.bool,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 160),
        )
        all_log_probs = tf.TensorArray(
            dtype=tf.float32,
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
        all_b2b_reward = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_combo_reward = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_spin_reward = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_height_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_hole_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_skyline_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_bumpy_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_death_penalty = tf.TensorArray(
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

        # Initialize pygame
        if render and not pygame.get_init():
            pygame.init()
            screen = pygame.display.set_mode((250, 600))
            pygame.display.set_caption("Tetris")
        else:
            screen = pygame.display.get_surface()

        time_step = self.env.reset()

        for t in range(self._num_steps):
            board = time_step.observation["board"]
            pieces = time_step.observation["pieces"]
            b2b_combo = time_step.observation["b2b_combo"]
            sequences = time_step.observation["sequences"]
            action_mask = tf.reduce_any(sequences != Keys.PAD, axis=-1)

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

            (
                action,
                log_prob,
                _,
            ) = self.p_model.predict((board, pieces, b2b_combo, action_mask))
            values = self.v_model.predict((board, pieces, b2b_combo))

            key_sequence = tf.gather(sequences, action, batch_dims=1)

            time_step = self.env.step(key_sequence)

            reward = time_step.reward
            attack = reward["attack"]
            clear = reward["clear"]
            b2b_reward = reward["b2b_reward"]
            combo_reward = reward["combo_reward"]
            spin_reward = reward["spin_reward"]
            height_penalty = reward["height_penalty"]
            hole_penalty = reward["hole_penalty"]
            skyline_penalty = reward["skyline_penalty"]
            bumpy_penalty = reward["bumpy_penalty"]
            death_penalty = reward["death_penalty"]

            dones = tf.cast(time_step.is_last(), tf.float32)[..., None]

            # Store the data
            all_boards = all_boards.write(t, board)
            all_pieces = all_pieces.write(t, pieces)
            all_b2b_combo = all_b2b_combo.write(t, b2b_combo)
            all_actions = all_actions.write(t, action)
            all_masks = all_masks.write(t, action_mask)
            all_log_probs = all_log_probs.write(t, log_prob)
            all_values = all_values.write(t, values)

            # Store the penalties and rewards
            all_attacks = all_attacks.write(t, attack)
            all_clears = all_clears.write(t, clear)
            all_b2b_reward = all_b2b_reward.write(t, b2b_reward)
            all_combo_reward = all_combo_reward.write(t, combo_reward)
            all_spin_reward = all_spin_reward.write(t, spin_reward)
            all_height_penalty = all_height_penalty.write(t, height_penalty)
            all_hole_penalty = all_hole_penalty.write(t, hole_penalty)
            all_skyline_penalty = all_skyline_penalty.write(t, skyline_penalty)
            all_bumpy_penalty = all_bumpy_penalty.write(t, bumpy_penalty)
            all_death_penalty = all_death_penalty.write(t, death_penalty)

            all_dones = all_dones.write(t, dones)

        all_boards = all_boards.stack()
        all_pieces = all_pieces.stack()
        all_b2b_combo = all_b2b_combo.stack()
        all_actions = all_actions.stack()
        all_masks = all_masks.stack()
        all_log_probs = all_log_probs.stack()
        all_values = all_values.stack()
        all_attacks = all_attacks.stack()
        all_clears = all_clears.stack()
        all_b2b_reward = all_b2b_reward.stack()
        all_combo_reward = all_combo_reward.stack()
        all_spin_reward = all_spin_reward.stack()
        all_height_penalty = all_height_penalty.stack()
        all_hole_penalty = all_hole_penalty.stack()
        all_skyline_penalty = all_skyline_penalty.stack()
        all_bumpy_penalty = all_bumpy_penalty.stack()
        all_death_penalty = all_death_penalty.stack()
        all_dones = all_dones.stack()

        return (
            all_boards,
            all_pieces,
            all_b2b_combo,
            all_actions,
            all_masks,
            all_log_probs,
            all_values,
            all_attacks,
            all_clears,
            all_b2b_reward,
            all_combo_reward,
            all_spin_reward,
            all_height_penalty,
            all_hole_penalty,
            all_skyline_penalty,
            all_bumpy_penalty,
            all_death_penalty,
            all_dones,
        )
