import gymnasium
from gymnasium import spaces
from .PyTetrisEnv import PyTetrisEnv
import numpy as np
import random
from typing import Dict, Optional


class PyTetris1v1Env(gymnasium.Env):
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
        max_steps: Optional[int],
        max_len: int,
        pathfinding: bool,
        seed: Optional[int],
        idx: int,
        gamma: float = 0.99,
        num_row_tiers: int = 2,
    ) -> None:
        self._max_holes = max_holes
        self._max_steps = max_steps
        self._max_len = max_len
        self._queue_size = queue_size
        self._num_row_tiers = num_row_tiers

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
        )
        self._env2 = PyTetrisEnv(
            queue_size=queue_size,
            max_holes=max_holes,
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
        )

        self._step_num = 0
        self._episode_ended = False

        num_sequences = 160 * num_row_tiers

        obs_spaces = {
            # Player 1 (training)
            "board": spaces.Box(0.0, 1.0, (24, 10, 1), np.float32),
            "vis_board": spaces.Box(0, 8, (24, 10, 1), np.int32),
            "pieces": spaces.Box(0, 7, (2 + queue_size,), np.int64),
            "b2b_combo_garbage": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "sequences": spaces.Box(0, 11, (num_sequences, max_len), np.int64),
            # Player 2 (opponent)
            "opp_board": spaces.Box(0.0, 1.0, (24, 10, 1), np.float32),
            "opp_pieces": spaces.Box(0, 7, (2 + queue_size,), np.int64),
            "opp_b2b_combo_garbage": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "opp_sequences": spaces.Box(0, 11, (num_sequences, max_len), np.int64),
        }
        self.observation_space = spaces.Dict(obs_spaces)
        # Two key sequences concatenated: player 1 (15) + player 2 (15).
        self.action_space = spaces.Box(0, 11, (30,), np.int64)

        print(f"Initialized 1v1 Env {idx}", flush=True)

    def _step_one_player(self, env, action):
        """Execute an action for one player. Returns
        (top_out, clear, attack, net_attack, is_spin, is_surge)."""
        pre_b2b = env._scorer._b2b
        (
            top_out,
            clear,
            attack,
            is_spin,
            board,
            vis_board,
            active_piece,
            hold_piece,
            queue,
        ) = env._execute_action(
            env._board,
            env._vis_board,
            env._active_piece,
            env._hold_piece,
            env._queue,
            action,
        )

        # Actual surge = a b2b chain (>=4) broken by this clear (Scorer.judge releases it).
        is_surge = clear > 0 and pre_b2b >= 4 and env._scorer._b2b == -1

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

        return top_out, clear, float(attack), float(net_attack), is_spin, is_surge

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._random = random.Random(seed)
            self._env1._seed = seed
            self._env2._seed = seed
        self._env1.reset()
        self._env2.reset()
        self._step_num = 0
        self._episode_ended = False
        self._random = random.Random(self._random.randint(0, 2**31))

        observation = self._create_1v1_observation()
        return observation, {}

    def step(self, combined_action: np.ndarray):
        self._step_num += 1

        action1 = combined_action[:15]
        action2 = combined_action[15:]

        # --- Execute both players' actions ---
        top_out1, clear1, attack1, net1, _, surge1 = self._step_one_player(
            self._env1, action1
        )
        top_out2, clear2, attack2, net2, _, surge2 = self._step_one_player(
            self._env2, action2
        )

        # --- Push existing garbage for non-clearing players ---
        # Push BEFORE injecting new attacks so incoming garbage always sits
        # in the queue for at least one turn (opponent can see and cancel it).
        garbage_pushed1 = False
        if clear1 == 0 and self._env1._garbage_queue:
            self._env1._tick_garbage_timers()
            self._env1._board, self._env1._vis_board, garbage_pushed1 = (
                self._env1._push_garbage_to_board(
                    self._env1._board, self._env1._vis_board
                )
            )

        if clear2 == 0 and self._env2._garbage_queue:
            self._env2._tick_garbage_timers()
            self._env2._board, self._env2._vis_board, _ = (
                self._env2._push_garbage_to_board(
                    self._env2._board, self._env2._vis_board
                )
            )

        # --- Inject net attacks as garbage into opponent (real surges split into waves) ---
        if net1 > 0:
            self._env2._receive_attack(int(net1), self._random.randint(0, 9), surge1)
        if net2 > 0:
            self._env1._receive_attack(int(net2), self._random.randint(0, 9), surge2)

        # --- Death checks ---
        p1_died = top_out1 or self._env1._is_top_out(self._env1._board)
        p2_died = top_out2 or self._env2._is_top_out(self._env2._board)

        h1, holes1, sky1, bump1 = self._env1._board_stats(self._env1._board)
        if self._max_holes is not None and holes1 > self._max_holes:
            p1_died = True

        h2, holes2, _, _ = self._env2._board_stats(self._env2._board)
        if self._max_holes is not None and holes2 > self._max_holes:
            p2_died = True

        # Diagnostic (not part of training reward)
        attack_reward = self._env1._attack_reward * net1

        # Terminal-only reward (zero-sum, equal magnitude)
        if p2_died and not p1_died:
            total_reward = self._win_reward
        elif p1_died and not p2_died:
            total_reward = -self._win_reward
        else:
            total_reward = 0.0

        # --- Fill queues ---
        self._env1._queue = self._env1._fill_queue(self._env1._queue)
        self._env2._queue = self._env2._fill_queue(self._env2._queue)

        # --- Observation ---
        observation = self._create_1v1_observation()

        p1_won = p2_died and not p1_died
        p1_lost = p1_died and not p2_died

        info = {
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

        terminated = bool(p1_died or p2_died)
        truncated = (
            not terminated
            and self._max_steps is not None
            and self._step_num >= self._max_steps
        )
        self._episode_ended = terminated or truncated

        return observation, float(total_reward), terminated, truncated, info

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
