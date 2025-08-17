from TetrisEnvs.PyTetrisEnv.PyTetrisEnv import PyTetrisEnv
import tensorflow as tf
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from TetrisEnvs.PyTetrisEnv.Moves import Convert
import pygame


class PyTetrisRunner:
    def __init__(
        self,
        queue_size,
        max_holes,
        max_height,
        max_steps,
        num_steps,
        num_envs,
        p_model,
        v_model,
        seed=123,
    ):
        self._queue_size = queue_size
        self._num_steps = num_steps
        self._num_envs = num_envs

        self.p_model = p_model
        self.v_model = v_model

        constructors = [
            lambda idx=i: PyTetrisEnv(
                queue_size=queue_size,
                max_holes=max_holes,
                max_height=max_height,
                max_steps=max_steps,
                seed=seed,
                idx=idx,
            )
            for i in range(num_envs)
        ]
        ppy_env = ParallelPyEnvironment(
            constructors, start_serially=True, blocking=False
        )
        self.env = TFPyEnvironment(ppy_env)

    def collect_trajectory(self, render=False):
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
        all_actions = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
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

            action, log_prob = self.p_model.predict((board, pieces))
            values = self.v_model.predict((board, pieces))

            key_sequence = tf.gather(Convert.to_sequence, action)

            time_step = self.env.step(key_sequence)

            reward = time_step.reward
            attack = reward["attack"]
            clear = reward["clear"]
            height_penalty = reward["height_penalty"]
            hole_penalty = reward["hole_penalty"]
            skyline_penalty = reward["skyline_penalty"]
            bumpy_penalty = reward["bumpy_penalty"]
            death_penalty = reward["death_penalty"]

            dones = tf.cast(time_step.is_last(), tf.float32)[..., None]

            # Store the data
            all_boards = all_boards.write(t, board)
            all_pieces = all_pieces.write(t, pieces)
            all_actions = all_actions.write(t, action)
            all_log_probs = all_log_probs.write(t, log_prob)
            all_values = all_values.write(t, values)

            # Store the penalties and rewards
            all_attacks = all_attacks.write(t, attack)
            all_clears = all_clears.write(t, clear)
            all_height_penalty = all_height_penalty.write(t, height_penalty)
            all_hole_penalty = all_hole_penalty.write(t, hole_penalty)
            all_skyline_penalty = all_skyline_penalty.write(t, skyline_penalty)
            all_bumpy_penalty = all_bumpy_penalty.write(t, bumpy_penalty)
            all_death_penalty = all_death_penalty.write(t, death_penalty)

            all_dones = all_dones.write(t, dones)

        all_boards = all_boards.stack()
        all_pieces = all_pieces.stack()
        all_actions = all_actions.stack()
        all_log_probs = all_log_probs.stack()
        all_values = all_values.stack()
        all_attacks = all_attacks.stack()
        all_clears = all_clears.stack()
        all_height_penalty = all_height_penalty.stack()
        all_hole_penalty = all_hole_penalty.stack()
        all_skyline_penalty = all_skyline_penalty.stack()
        all_bumpy_penalty = all_bumpy_penalty.stack()
        all_death_penalty = all_death_penalty.stack()
        all_dones = all_dones.stack()

        return (
            all_boards,
            all_pieces,
            all_actions,
            all_log_probs,
            all_values,
            all_attacks,
            all_clears,
            all_height_penalty,
            all_hole_penalty,
            all_skyline_penalty,
            all_bumpy_penalty,
            all_death_penalty,
            all_dones,
        )
