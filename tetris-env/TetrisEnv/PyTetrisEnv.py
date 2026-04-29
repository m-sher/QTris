from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from .RotationSystem import RotationSystem
from .Scorer import Scorer
from .Pieces import Piece, PieceType
from .Moves import Moves, Keys
from .TetrioRandom import TetrioRNG
from .CKeySequences import CKeySequenceFinder
from .CHoleFinder import CHoleFinder
from .helpers import overlaps
import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Optional


class PyTetrisEnv(py_environment.PyEnvironment):
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
        garbage_chance: float = 0.0,
        garbage_min: int = 0,
        garbage_max: int = 0,
        garbage_push_delay: int = 1,
        gamma: float = 0.99,
        auto_push_garbage: bool = True,
        auto_fill_queue: bool = True,
        num_row_tiers: int = 1,
        use_shaping: bool = True,
    ) -> None:
        self._attack_reward = 1.0
        self._b2b_coef = 2.0
        self._surge_coef = 1.0
        self._combo_coef = 0.5
        self._safety_coef = 10.0
        self._hole_coef = 1.0
        self._skyline_coef = 0.1
        self._bumpy_coef = 0.2
        self._death_penalty = -100.0

        self._max_holes = max_holes
        self._max_height = max_height
        self._max_steps = max_steps
        self._max_len = max_len
        self._pathfinding = pathfinding
        self._num_row_tiers = num_row_tiers

        self._garbage_chance = garbage_chance
        self._garbage_min = garbage_min
        self._garbage_max = garbage_max
        self._garbage_push_delay = garbage_push_delay

        self._gamma = gamma
        self._auto_push_garbage = auto_push_garbage
        self._auto_fill_queue = auto_fill_queue
        self._use_shaping = use_shaping

        self._seed = seed

        self._random = random.Random(seed)
        self._tetrio_rng = TetrioRNG(seed)

        self._board = np.zeros((24, 10), dtype=np.float32)
        self._vis_board = np.zeros((24, 10), dtype=np.int32)

        self._rotation_system = RotationSystem()
        self._key_sequence_finder = CKeySequenceFinder(self._rotation_system, num_row_tiers=num_row_tiers)
        self._hole_finder = CHoleFinder()
        self._scorer = Scorer()

        self._step_num = 0

        self._last_phi = 0

        self._hold_piece = PieceType.N

        self._queue_size = queue_size
        self._next_bag = self._tetrio_rng.next_bag()

        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._queue = self._fill_queue([])

        # Initialize garbage queue - stores (num_rows, empty_column, push_timer)
        # push_timer counts down non-clear steps until the garbage is pushed.
        self._garbage_queue: List[Tuple[int, int, int]] = []

        self._episode_ended = False

        self._observation_spec = {
            "board": array_spec.BoundedArraySpec(
                shape=(24, 10, 1),
                dtype=np.float32,
                minimum=0.0,
                maximum=1.0,
                name="board",
            ),
            "vis_board": array_spec.BoundedArraySpec(
                shape=(24, 10, 1),
                dtype=np.int32,
                minimum=0.0,
                maximum=8.0,
                name="vis_board",
            ),
            "pieces": array_spec.BoundedArraySpec(
                shape=(2 + queue_size,),
                dtype=np.int64,
                minimum=0,
                maximum=7,
                name="pieces",
            ),
            "b2b_combo_garbage": array_spec.ArraySpec(
                shape=(3,), dtype=np.float32, name="b2b_combo"
            ),
            "sequences": array_spec.ArraySpec(
                shape=(160 * num_row_tiers, max_len), dtype=np.int64, name="sequences"
            ),
        }

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(15,), dtype=np.int64, minimum=0, maximum=11, name="key_sequence"
        )

        self._reward_spec = {
            "attack": array_spec.ArraySpec(shape=(), dtype=np.float32, name="attack"),
            "clear": array_spec.ArraySpec(shape=(), dtype=np.float32, name="clear"),
            "attack_reward": array_spec.ArraySpec(
                shape=(), dtype=np.float32, name="attack_reward"
            ),
            "total_reward": array_spec.ArraySpec(
                shape=(), dtype=np.float32, name="total_reward"
            ),
            "garbage_pushed": array_spec.ArraySpec(
                shape=(), dtype=np.float32, name="garbage_pushed"
            ),
        }

        print(f"Initialized Env {idx}", flush=True)

    def action_spec(self) -> array_spec.BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> Dict[str, array_spec.ArraySpec]:
        return self._observation_spec

    def reward_spec(self) -> Dict[str, array_spec.ArraySpec]:
        return self._reward_spec

    def _calculate_potential(self, b2b: int, combo: int, height: float, holes: float, skyline: float, bumpiness: float) -> float:
        
        b2b_level = max(0.0, b2b)
        surge_lines = b2b_level if b2b_level >= 4 else 0
        combo_level = max(0.0, combo)

        # Main goal (is to blow up and act like I don't know nobody ackackackackack)
        phi_target = (
            (self._b2b_coef * np.log(1 + b2b_level)) +
            (self._surge_coef * 1.15 ** surge_lines - 1) +
            (self._combo_coef * np.log(1 + combo_level))
        )

        # Survival
        height_norm = height / self._max_height
        phi_safety = self._safety_coef * (height_norm ** 2)

        # Cleanliness
        phi_clean = (self._hole_coef * holes) + (self._skyline_coef * skyline) + (self._bumpy_coef * bumpiness)

        return phi_target - phi_safety - phi_clean

    def _reset(self) -> ts.TimeStep:
        self._seed = (self._seed + 1) if self._seed is not None else None

        self._random = random.Random(self._seed)
        self._tetrio_rng._seed = self._seed
        self._tetrio_rng.reset()

        self._board[:] = 0.0
        self._vis_board[:] = 0

        self._scorer.reset()

        self._step_num = 0

        self._last_phi = 0

        self._hold_piece = PieceType.N

        self._next_bag = self._tetrio_rng.next_bag()

        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._queue = self._fill_queue([])

        # Reset garbage queue
        self._garbage_queue = []

        self._episode_ended = False

        observation = self._create_observation()

        return ts.restart(observation=observation, reward_spec=self._reward_spec)

    def _step(self, key_sequence: np.ndarray) -> ts.TimeStep:
        """
        `_lock_piece` does not move piece to the bottom, and only tries
        locking at the current location. Action sequences all already end in
        hard drop."
        """
        self._step_num += 1

        if self._episode_ended:
            return self.reset()

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
        ) = self._execute_action(
            self._board,
            self._vis_board,
            self._active_piece,
            self._hold_piece,
            self._queue,
            key_sequence,
        )

        if attack > 0:
            self._remove_attack_from_garbage_queue(attack)

        garbage_pushed = False
        if self._auto_push_garbage and clear == 0:  # No lines were cleared
            self._tick_garbage_timers()
            board, vis_board, garbage_pushed = self._push_garbage_to_board(board, vis_board)

        # Check if new garbage should be added to queue
        self._add_to_garbage_queue()

        # delay=0: push newly generated garbage immediately (same step)
        if self._auto_push_garbage and self._garbage_push_delay == 0:
            while self._garbage_queue:
                board, vis_board, pushed = self._push_garbage_to_board(board, vis_board)
                if not pushed:
                    break
                garbage_pushed = True

        # Get board stats and compute supplementary rewards AFTER garbage
        height_val, holes_val, skyline_val, bumpy_val = self._board_stats(board)

        b2b_val = self._scorer._b2b
        combo_val = self._scorer._combo

        attack_reward = self._attack_reward * attack

        if self._use_shaping:
            current_phi = self._calculate_potential(b2b_val, combo_val, height_val, holes_val, skyline_val, bumpy_val)
            shaping_reward = (self._gamma * current_phi) - self._last_phi
        else:
            current_phi = 0.0
            shaping_reward = 0.0

        exceeded_holes = (
            holes_val > self._max_holes if self._max_holes is not None else False
        )

        # Check for top-out caused by garbage
        garbage_top_out = np.any(board[: 24 - self._max_height] != 0.0)

        died = top_out or exceeded_holes or garbage_top_out
 
        total_reward = attack_reward + shaping_reward + (self._death_penalty if died else 0.0)

        if self._auto_fill_queue:
            queue = self._fill_queue(queue)

        # Update state
        self._board = board
        self._vis_board = vis_board
        self._active_piece = active_piece
        self._hold_piece = hold_piece
        self._queue = queue

        self._last_phi = current_phi

        observation = self._create_observation()

        reward = {
            "attack": np.array(attack),
            "clear": np.array(clear),
            "attack_reward": np.array(attack_reward),
            "total_reward": np.array(total_reward),
            "garbage_pushed": np.array(float(garbage_pushed)),
        }

        self._episode_ended = died or (
            False if not self._max_steps else self._step_num >= self._max_steps
        )

        if self._episode_ended:
            return ts.termination(observation=observation, reward=reward)
        else:
            return ts.transition(observation=observation, reward=reward)

    def _execute_action(
        self,
        board: np.ndarray,
        vis_board: np.ndarray,
        active_piece: Piece,
        hold_piece: PieceType,
        queue: List[PieceType],
        key_sequence: np.ndarray,
    ) -> Tuple[
        bool,
        int,
        float,
        bool,
        np.ndarray,
        np.ndarray,
        Piece,
        PieceType,
        List[PieceType],
    ]:
        # Avoid modifying original state
        board = copy.deepcopy(board)
        active_piece = copy.deepcopy(active_piece)
        hold_piece = copy.deepcopy(hold_piece)
        queue = copy.deepcopy(queue)

        clear = 0
        top_out = False
        next_piece = None
        attack = 0
        can_hold = True
        for key in key_sequence:
            if key == Keys.HOLD:
                can_hold, active_piece, hold_piece, queue = self._try_hold(
                    can_hold, active_piece, hold_piece, queue
                )
            elif key in Keys.key_vectors.keys():
                key_vector = Keys.key_vectors[key]
                active_piece = self._try_key_vector(key_vector, active_piece, board)

                if key == Keys.HARD_DROP:
                    clear, top_out, next_piece, next_board, next_vis_board, queue = (
                        self._lock_piece(active_piece, board, vis_board, queue)
                    )

                    attack, is_spin = self._scorer.judge(active_piece, board, next_board, clear)

        return (
            top_out,
            clear,
            attack,
            is_spin,
            next_board,
            next_vis_board,
            next_piece,
            hold_piece,
            queue,
        )

    def _spawn_piece(self, piece_type: PieceType) -> Piece:
        # All pieces spawn 3 cells from the left on a default board
        # For the O piece, this is actually 4 cells including the padding
        spawn_loc = np.array([0, 3], np.int32)
        cells = self._rotation_system.orientations[piece_type][0]

        return Piece(piece_type=piece_type, loc=spawn_loc, r=0, cells=cells)

    def _create_observation(self) -> Dict[str, np.ndarray]:
        pieces = [self._active_piece.piece_type, self._hold_piece] + self._queue
        pieces = np.array([piece.value for piece in pieces], dtype=np.int64)

        total_garbage = self._get_total_garbage()
        stats = np.array([self._scorer._b2b, self._scorer._combo, total_garbage], dtype=np.float32)

        if self._pathfinding:
            non_hold_sequences = self._key_sequence_finder.find_all(
                board=self._board,
                piece=self._active_piece,
                max_len=self._max_len,
                is_hold=False,
            )

            hold_sequences = self._key_sequence_finder.find_all(
                board=self._board,
                piece=(
                    self._spawn_piece(self._hold_piece)
                    if self._hold_piece != PieceType.N
                    else self._spawn_piece(self._queue[0])
                ),
                max_len=self._max_len,
                is_hold=True,
            )

            sequences = np.concatenate([non_hold_sequences, hold_sequences], axis=0)
        else:
            sequences = np.zeros((160 * self._num_row_tiers, self._max_len), dtype=np.int64)

        observation = {
            "board": self._board[..., None],
            "vis_board": self._vis_board[..., None],
            "pieces": pieces,
            "b2b_combo_garbage": stats,
            "sequences": sequences,
        }

        return observation

    def _convert_to_keys(self, action: Dict[str, int]) -> List[int]:
        hold = Moves._holds[action["hold"]]
        standard = Moves._standards[action["standard"]]
        spin = Moves._spins[action["spin"]]

        key_sequence = hold + standard + spin + [Keys.HARD_DROP]

        return key_sequence

    def _try_key_vector(
        self, key_vector: np.ndarray, active_piece: Piece, board: np.ndarray
    ) -> Piece:
        # Key vector is delta [row, column, rotation]

        try_delta_loc = key_vector[:-1]
        try_delta_r = key_vector[-1]

        if try_delta_r:
            active_piece = self._rotate(
                try_delta_r, active_piece=active_piece, board=board
            )
        else:
            active_piece = self._move(
                try_delta_loc, active_piece=active_piece, board=board
            )

        return active_piece

    def _rotate(
        self, try_delta_r: int, active_piece: Piece, board: np.ndarray
    ) -> Piece:
        new_r = (active_piece.r + try_delta_r) % 4
        cells = self._rotation_system.orientations[active_piece.piece_type][new_r]

        if not overlaps(cells=cells, loc=active_piece.loc, board=board):
            # Applying rotation doesn't overlap, so do it
            active_piece.r = new_r
            active_piece.delta_r = try_delta_r

            active_piece.delta_loc = np.zeros((2,), dtype=np.int32)
            active_piece.cells = cells

        else:
            # Overlaps, so try kicks
            kick_table = (
                self._rotation_system.i_kicks
                if active_piece.piece_type == PieceType.I
                else self._rotation_system.kicks
            )
            self._rotation_system.kick_piece(
                kick_table, active_piece, cells, new_r, try_delta_r, board
            )

        return active_piece

    def _move(
        self, try_delta_loc: np.ndarray, active_piece: Piece, board: np.ndarray
    ) -> Piece:
        # Vertical movement
        if try_delta_loc[0]:
            # -1 = up (NEVER)
            # +1 = down
            direction = np.sign(try_delta_loc[0])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_y in range(0, try_delta_loc[0] + direction, direction):
                new_delta_loc = np.array([delta_y, 0], dtype=np.int32)
                if overlaps(
                    cells=active_piece.cells,
                    loc=active_piece.loc + new_delta_loc,
                    board=board,
                ):
                    break
                else:
                    delta_loc = new_delta_loc

        # Horizontal movement
        elif try_delta_loc[1]:
            # -1 = left
            # +1 = right
            direction = np.sign(try_delta_loc[1])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_x in range(0, try_delta_loc[1] + direction, direction):
                new_delta_loc = np.array([0, delta_x], dtype=np.int32)
                if overlaps(
                    cells=active_piece.cells,
                    loc=active_piece.loc + new_delta_loc,
                    board=board,
                ):
                    break
                else:
                    delta_loc = new_delta_loc

        else:
            # Delta would have to be [0, 0] to get here
            raise ValueError(f"Should never reach this! Delta: {try_delta_loc}")

        active_piece.loc += delta_loc
        active_piece.delta_loc = delta_loc

        if np.any(delta_loc != 0):
            active_piece.delta_r = 0

        return active_piece

    def _try_hold(
        self,
        can_hold: bool,
        active_piece: Piece,
        hold_piece: PieceType,
        queue: List[PieceType],
    ) -> Tuple[bool, Piece, PieceType, List[PieceType]]:
        # Using `_can_hold` is unnecessary for this implementation
        # since only valid action sequences exist and none include pressing
        # the hold key twice. This is included in case of future changes.
        if can_hold:
            if hold_piece == PieceType.N:
                # No piece held, so this is the first time holding
                hold_piece = active_piece.piece_type
                active_piece = self._spawn_piece(queue.pop(0))
            else:
                active_piece, hold_piece = (
                    self._spawn_piece(hold_piece),
                    active_piece.piece_type,
                )
            can_hold = False

        return can_hold, active_piece, hold_piece, queue

    def _lock_piece(
        self,
        active_piece: Piece,
        board: np.ndarray,
        vis_board: np.ndarray,
        queue: List[PieceType],
    ) -> Tuple[int, bool, Piece, np.ndarray, np.ndarray, List[PieceType]]:
        # DOES NOT MOVE PIECE TO THE BOTTOM
        # This method assumes the piece is already in a settled position.
        board = copy.deepcopy(board)
        vis_board = copy.deepcopy(vis_board)

        cell_coords = active_piece.cells + active_piece.loc

        board[cell_coords[:, 0], cell_coords[:, 1]] = 1.0
        vis_board[cell_coords[:, 0], cell_coords[:, 1]] = active_piece.piece_type.value

        board, vis_board, clears = self._clear_lines(board, vis_board)

        active_piece = self._spawn_piece(queue.pop(0))

        top_out = np.any(board[: 24 - self._max_height] != 0.0)

        return clears, top_out, active_piece, board, vis_board, queue

    def _fill_queue(self, queue: List[PieceType]) -> List[PieceType]:
        while len(queue) < self._queue_size:
            queue.append(self._next_bag.pop(0))
            if len(self._next_bag) == 0:
                self._next_bag = self._tetrio_rng.next_bag()
        return queue

    def _clear_lines(
        self, board: np.ndarray, vis_board: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        clears = 0
        for i, row in enumerate(board):
            if np.all(row != 0.0):
                clears += 1
                board[0] = 0.0
                board[1 : i + 1] = board[:i]

                vis_board[0] = 0
                vis_board[1 : i + 1] = vis_board[:i]
        return board, vis_board, clears

    def _add_to_garbage_queue(self) -> None:
        """Generate garbage and add it to the garbage queue based on chance."""
        if self._garbage_chance <= 0.0 or self._garbage_max <= 0:
            return

        if self._random.random() > self._garbage_chance:
            return

        if self._garbage_min == self._garbage_max:
            num_garbage_rows = self._garbage_min
        else:
            num_garbage_rows = self._random.randint(
                self._garbage_min, self._garbage_max
            )

        if num_garbage_rows <= 0:
            return

        empty_column = self._random.randint(0, 9)

        # Add garbage instance to queue with push delay timer
        self._garbage_queue.append((num_garbage_rows, empty_column, self._garbage_push_delay))

    def _tick_garbage_timers(self) -> None:
        """Decrement push_timer on all queued garbage (called on non-clear steps)."""
        for i in range(len(self._garbage_queue)):
            rows, col, timer = self._garbage_queue[i]
            if timer > 0:
                self._garbage_queue[i] = (rows, col, timer - 1)

    def _push_garbage_to_board(
        self, board: np.ndarray, vis_board: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Push the next garbage from the queue to the board if its timer expired."""
        if not self._garbage_queue:
            return board, vis_board, False

        # Only push if front item's timer has reached 0
        if self._garbage_queue[0][2] > 0:
            return board, vis_board, False

        num_garbage_rows, empty_column, _ = self._garbage_queue.pop(0)

        garbage_rows = np.ones((num_garbage_rows, 10), dtype=np.float32)
        garbage_rows[:, empty_column] = 0.0

        board = np.concatenate([board[num_garbage_rows:], garbage_rows], axis=0)
        vis_board = np.concatenate(
            [vis_board[num_garbage_rows:], garbage_rows * PieceType.G.value], axis=0
        ).astype(np.int32)

        return board, vis_board, True

    def _remove_attack_from_garbage_queue(self, attack_amount: float) -> None:
        """Remove garbage from queue based on attack sent."""
        lines_to_remove = int(attack_amount)

        while lines_to_remove > 0 and self._garbage_queue:
            num_garbage_rows, empty_column, timer = self._garbage_queue[0]

            if num_garbage_rows <= lines_to_remove:
                # Remove entire garbage instance
                lines_to_remove -= num_garbage_rows
                self._garbage_queue.pop(0)
            else:
                # Partially reduce garbage instance
                self._garbage_queue[0] = (num_garbage_rows - lines_to_remove, empty_column, timer)
                lines_to_remove = 0

    def _get_total_garbage(self) -> int:
        return sum(rows for rows, _, _ in self._garbage_queue)

    def _get_heights(self, board: np.ndarray) -> np.ndarray:
        # Get heights of each column in the board
        height_matrix = np.arange(board.shape[0], 0, -1)[..., None]
        heights = np.max(board * height_matrix, axis=0)

        return heights

    def _get_holes(self, board: np.ndarray) -> int:
        # Count enclosed holes in the board
        return self._hole_finder.count_holes(board)

    def _get_skyline(self, heights: np.ndarray) -> float:
        # Computes the difference between the hightest columns and lowest columns
        # For a board with width 10, the skyline is the difference between the
        # sum of the 4 highest columns and the sum of the next 4 highest columns
        sorted_heights = np.sort(heights)
        skyline = np.sum(sorted_heights[-4:]) - np.sum(sorted_heights[2:-4])
        return skyline

    def _get_bumpy(self, heights: np.ndarray) -> np.ndarray:
        # Get bumpiness of the board
        bumpy = np.abs(heights[:-1] - heights[1:])
        return bumpy

    def _board_stats(self, board: np.ndarray) -> Tuple[float, float, float, float]:
        # Get total heights of the board
        heights = self._get_heights(board)
        height_val = np.max(heights)

        # Get number of holes in the board
        holes_val = self._get_holes(board)

        # Get skyline of the board
        skyline_val = self._get_skyline(heights)

        # Get bumpiness of the board
        bumpy = self._get_bumpy(heights)
        bumpy_val = np.sum(bumpy)

        # Return heights, holes, and bumpy
        return height_val, holes_val, skyline_val, bumpy_val
