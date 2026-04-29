from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from .PyTetrisEnv import PyTetrisEnv
from .Pieces import PieceType
import numpy as np
import random
from typing import Dict, Optional


class PyTetris1v1Env(py_environment.PyEnvironment):
    """1v1 Tetris environment for self-play training.

    Wraps two PyTetrisEnv instances with garbage_chance=0 and auto_push_garbage=False.
    Attack from each player is sent as garbage to the opponent after cancellation.
    Only player 1's rewards are computed (player 2 is the opponent).
    """

    _win_reward = 100.0

    def __init__(
        self,
        queue_size: int,
        max_holes: Optional[int],
        max_height: int,
        max_steps: Optional[int],
        max_len: int,
        pathfinding: bool,
        seed: Optional[int],
        idx: int,
        gamma: float = 0.99,
        num_row_tiers: int = 2,
        use_shaping: bool = False,
        b2b_gap_coef: float = 0.0,
    ) -> None:
        self._max_holes = max_holes
        self._max_height = max_height
        self._max_steps = max_steps
        self._max_len = max_len
        self._queue_size = queue_size
        self._num_row_tiers = num_row_tiers
        self._gamma = gamma
        self._use_shaping = use_shaping
        self._b2b_gap_coef = b2b_gap_coef
        self._last_b2b_gap_phi = 0.0

        # Ensure a non-None seed so both sides stay in sync across resets
        if seed is None:
            seed = random.randint(0, 2**31)

        # Separate RNG for garbage hole column selection
        self._random = random.Random(seed)

        # Player 1 (training agent) and Player 2 (opponent)
        # Both have no random garbage — garbage comes from opponent attacks
        self._env1 = PyTetrisEnv(
            queue_size=queue_size,
            max_holes=max_holes,
            max_height=max_height,
            max_steps=None,  # We handle max_steps at the 1v1 level
            max_len=max_len,
            pathfinding=pathfinding,
            seed=seed,
            idx=idx,
            garbage_chance=0.0,
            garbage_min=0,
            garbage_max=0,
            gamma=gamma,
            auto_push_garbage=False,
            auto_fill_queue=False,
            num_row_tiers=num_row_tiers,
            use_shaping=use_shaping,
        )
        self._env2 = PyTetrisEnv(
            queue_size=queue_size,
            max_holes=max_holes,
            max_height=max_height,
            max_steps=None,
            max_len=max_len,
            pathfinding=pathfinding,
            seed=seed,
            idx=idx,
            garbage_chance=0.0,
            garbage_min=0,
            garbage_max=0,
            gamma=gamma,
            auto_push_garbage=False,
            auto_fill_queue=False,
            num_row_tiers=num_row_tiers,
            use_shaping=use_shaping,
        )

        self._step_num = 0
        self._episode_ended = False

        num_sequences = 160 * num_row_tiers

        self._observation_spec = {
            # Player 1 (training)
            "board": array_spec.BoundedArraySpec(
                shape=(24, 10, 1), dtype=np.float32, minimum=0.0, maximum=1.0, name="board",
            ),
            "vis_board": array_spec.BoundedArraySpec(
                shape=(24, 10, 1), dtype=np.int32, minimum=0, maximum=8, name="vis_board",
            ),
            "pieces": array_spec.BoundedArraySpec(
                shape=(2 + queue_size,), dtype=np.int64, minimum=0, maximum=7, name="pieces",
            ),
            "b2b_combo_garbage": array_spec.ArraySpec(
                shape=(3,), dtype=np.float32, name="b2b_combo_garbage",
            ),
            "sequences": array_spec.ArraySpec(
                shape=(num_sequences, max_len), dtype=np.int64, name="sequences",
            ),
            # Player 2 (opponent)
            "opp_board": array_spec.BoundedArraySpec(
                shape=(24, 10, 1), dtype=np.float32, minimum=0.0, maximum=1.0, name="opp_board",
            ),
            "opp_pieces": array_spec.BoundedArraySpec(
                shape=(2 + queue_size,), dtype=np.int64, minimum=0, maximum=7, name="opp_pieces",
            ),
            "opp_b2b_combo_garbage": array_spec.ArraySpec(
                shape=(3,), dtype=np.float32, name="opp_b2b_combo_garbage",
            ),
            "opp_sequences": array_spec.ArraySpec(
                shape=(num_sequences, max_len), dtype=np.int64, name="opp_sequences",
            ),
        }

        # Two key sequences concatenated: player 1 (15) + player 2 (15)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(30,), dtype=np.int64, minimum=0, maximum=11, name="key_sequences",
        )

        self._reward_spec = {
            "attack": array_spec.ArraySpec(shape=(), dtype=np.float32, name="attack"),
            "net_attack": array_spec.ArraySpec(shape=(), dtype=np.float32, name="net_attack"),
            "clear": array_spec.ArraySpec(shape=(), dtype=np.float32, name="clear"),
            "attack_reward": array_spec.ArraySpec(shape=(), dtype=np.float32, name="attack_reward"),
            "total_reward": array_spec.ArraySpec(shape=(), dtype=np.float32, name="total_reward"),
            "garbage_pushed": array_spec.ArraySpec(shape=(), dtype=np.float32, name="garbage_pushed"),
            "win": array_spec.ArraySpec(shape=(), dtype=np.float32, name="win"),
            "loss": array_spec.ArraySpec(shape=(), dtype=np.float32, name="loss"),
            "opp_attack": array_spec.ArraySpec(shape=(), dtype=np.float32, name="opp_attack"),
            "opp_clear": array_spec.ArraySpec(shape=(), dtype=np.float32, name="opp_clear"),
        }

        print(f"Initialized 1v1 Env {idx}", flush=True)

    def action_spec(self) -> array_spec.BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> Dict[str, array_spec.ArraySpec]:
        return self._observation_spec

    def reward_spec(self) -> Dict[str, array_spec.ArraySpec]:
        return self._reward_spec

    def _step_one_player(self, env, action):
        """Execute an action for one player. Returns (top_out, clear, attack, net_attack, is_spin)."""
        (
            top_out, clear, attack, is_spin,
            board, vis_board, active_piece, hold_piece, queue,
        ) = env._execute_action(
            env._board, env._vis_board, env._active_piece,
            env._hold_piece, env._queue, action,
        )

        # Cancel own pending garbage with outgoing attack
        pending_before = env._get_total_garbage()
        if attack > 0:
            env._remove_attack_from_garbage_queue(attack)
        pending_after = env._get_total_garbage()
        net_attack = attack - (pending_before - pending_after)

        # Update env state
        env._board = board
        env._vis_board = vis_board
        env._active_piece = active_piece
        env._hold_piece = hold_piece
        env._queue = queue

        return top_out, clear, float(attack), float(net_attack), is_spin

    def _reset(self) -> ts.TimeStep:
        self._env1._reset()
        self._env2._reset()
        self._step_num = 0
        self._episode_ended = False
        self._last_b2b_gap_phi = 0.0
        self._random = random.Random(self._random.randint(0, 2**31))

        observation = self._create_1v1_observation()
        return ts.restart(observation=observation, reward_spec=self._reward_spec)

    def _step(self, combined_action: np.ndarray) -> ts.TimeStep:
        self._step_num += 1

        if self._episode_ended:
            return self.reset()

        action1 = combined_action[:15]
        action2 = combined_action[15:]

        # --- Execute both players' actions ---
        top_out1, clear1, attack1, net1, _ = self._step_one_player(self._env1, action1)
        top_out2, clear2, attack2, net2, _ = self._step_one_player(self._env2, action2)

        # --- Push existing garbage for non-clearing players ---
        # Push BEFORE injecting new attacks so incoming garbage always sits
        # in the queue for at least one turn (opponent can see and cancel it).
        garbage_pushed1 = False
        if clear1 == 0 and self._env1._garbage_queue:
            self._env1._tick_garbage_timers()
            self._env1._board, self._env1._vis_board, garbage_pushed1 = (
                self._env1._push_garbage_to_board(self._env1._board, self._env1._vis_board)
            )

        if clear2 == 0 and self._env2._garbage_queue:
            self._env2._tick_garbage_timers()
            self._env2._board, self._env2._vis_board, _ = (
                self._env2._push_garbage_to_board(self._env2._board, self._env2._vis_board)
            )

        # --- Inject net attacks as garbage into opponent ---
        if net1 > 0:
            col = self._random.randint(0, 9)
            self._env2._garbage_queue.append((int(net1), col, self._env2._garbage_push_delay))
        if net2 > 0:
            col = self._random.randint(0, 9)
            self._env1._garbage_queue.append((int(net2), col, self._env1._garbage_push_delay))

        # --- Death checks ---
        p1_died = top_out1 or np.any(self._env1._board[:24 - self._max_height] != 0.0)
        p2_died = top_out2 or np.any(self._env2._board[:24 - self._max_height] != 0.0)

        h1, holes1, sky1, bump1 = self._env1._board_stats(self._env1._board)
        if self._max_holes is not None and holes1 > self._max_holes:
            p1_died = True

        h2, holes2, _, _ = self._env2._board_stats(self._env2._board)
        if self._max_holes is not None and holes2 > self._max_holes:
            p2_died = True

        # --- Reward for player 1 ---
        b2b_level = max(0, self._env1._scorer._b2b)
        b2b_mult = 1.0 + 0.5 * b2b_level + 2.0 * max(0, b2b_level - 3)
        attack_reward = self._env1._attack_reward * net1 * b2b_mult

        if self._use_shaping:
            b2b_val = self._env1._scorer._b2b
            combo_val = self._env1._scorer._combo
            current_phi = self._env1._calculate_potential(b2b_val, combo_val, h1, holes1, sky1, bump1)
            shaping_reward = (self._gamma * current_phi) - self._env1._last_phi
            self._env1._last_phi = current_phi
        else:
            shaping_reward = 0.0

        # --- B2B gap potential-based shaping ---
        if self._b2b_gap_coef > 0.0:
            b2b_gap = self._env1._scorer._b2b - self._env2._scorer._b2b
            b2b_gap_phi = self._b2b_gap_coef * b2b_gap
            b2b_gap_reward = self._gamma * b2b_gap_phi - self._last_b2b_gap_phi
            self._last_b2b_gap_phi = b2b_gap_phi
        else:
            b2b_gap_reward = 0.0

        death_penalty = self._env1._death_penalty if p1_died else 0.0
        win_reward = self._win_reward if p2_died and not p1_died else 0.0

        total_reward = attack_reward + shaping_reward + b2b_gap_reward + death_penalty + win_reward

        # --- Fill queues ---
        self._env1._queue = self._env1._fill_queue(self._env1._queue)
        self._env2._queue = self._env2._fill_queue(self._env2._queue)

        # --- Observation ---
        observation = self._create_1v1_observation()

        p1_won = p2_died and not p1_died
        p1_lost = p1_died and not p2_died

        reward = {
            "attack": np.float32(attack1),
            "net_attack": np.float32(net1),
            "clear": np.float32(clear1),
            "attack_reward": np.float32(attack_reward),
            "total_reward": np.float32(total_reward),
            "garbage_pushed": np.float32(garbage_pushed1),
            "win": np.float32(p1_won),
            "loss": np.float32(p1_lost),
            "opp_attack": np.float32(attack2),
            "opp_clear": np.float32(clear2),
        }

        self._episode_ended = p1_died or p2_died or (
            self._max_steps is not None and self._step_num >= self._max_steps
        )

        if self._episode_ended:
            return ts.termination(observation=observation, reward=reward)
        else:
            return ts.transition(observation=observation, reward=reward)

    def _create_1v1_observation(self) -> Dict[str, np.ndarray]:
        obs1 = self._env1._create_observation()
        obs2 = self._env2._create_observation()

        return {
            "board": obs1["board"],
            "vis_board": obs1["vis_board"],
            "pieces": obs1["pieces"],
            "b2b_combo_garbage": obs1["b2b_combo_garbage"],
            "sequences": obs1["sequences"],
            "opp_board": obs2["board"],
            "opp_pieces": obs2["pieces"],
            "opp_b2b_combo_garbage": obs2["b2b_combo_garbage"],
            "opp_sequences": obs2["sequences"],
        }
