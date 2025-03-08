from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from .RotationSystem import RotationSystem
from .Scorer import Scorer
from .Pieces import Piece, PieceType
from .Moves import Moves, Keys
import numpy as np
import random

class TetrisPyEnv(py_environment.PyEnvironment):

    def __init__(self, queue_size, seed):

        self._seed = seed

        self._random = random.Random(seed)

        self._queue_size = queue_size

        self._board = np.zeros((24, 10), dtype=np.float32)

        self._rotation_system = RotationSystem(board=self._board)

        self._scorer = Scorer()

        self._valid_pieces = [piece for piece in PieceType if piece is not PieceType.N]

        self._reset()

        self._observation_spec = {
            'board': array_spec.BoundedArraySpec(
                shape=(24, 10), dtype=np.float32, minimum=0.0, maximum=1.0, name='board'),
            'pieces': array_spec.BoundedArraySpec(
                shape=(2 + queue_size,), dtype=np.int32, minimum=0, maximum=8, name='pieces')
        }

        self._action_spec = {
            'hold': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=1, name='hold'),
            'standard': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=36, name='standard'),
            'spin': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=7, name='spin')
        }

        print("Initialized Env", flush=True)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        self._random = random.Random(self._seed + 1)

        self._board[:] = 0
        self._hold_piece = PieceType.N
        self._can_hold = True

        self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._queue = []
        self._fill_queue()

        self._scorer.reset()

        self._clears = 0

        self._episode_ended = False

        observation = self._create_observation()

        return ts.restart(observation)

    def _step(self, action: dict[str, int]):
        # `_lock_piece` does not move piece to the bottom, and only tries
        # locking at the current location. Action sequences all already end in
        # hard drop.
        if self._episode_ended:
            return self.reset()

        key_sequence = self._convert_to_keys(action)

        for key in key_sequence:
            if key == Keys.HOLD:
                self._try_hold()
            elif key in Keys.key_vectors.keys():
                key_vector = Keys.key_vectors[key]
                self._try_key_vector(key_vector)

                if key == Keys.HARD_DROP:
                    self._episode_ended = not self._lock_piece()

            reward = self._scorer.judge(self._active_piece, self._board, key, self._clears)

        self._fill_queue()
        self._can_hold = True

        observation = self._create_observation()

        if self._episode_ended:
            return ts.termination(observation, reward=reward)
        else:
            return ts.transition(observation, reward=reward)

    def _spawn_piece(self, piece_type: PieceType) -> Piece:
        # All pieces spawn 3 cells from the left on a default board
        # For the O piece, this is actually 4 cells including the padding
        spawn_loc = np.array([0, 3], np.int32)
        cells = self._rotation_system.orientations[piece_type][0]

        return Piece(piece_type=piece_type,
                     loc=spawn_loc,
                     r=0,
                     cells=cells)

    def _create_observation(self):
        pieces = [self._active_piece.piece_type, self._hold_piece] + self._queue
        pieces = np.array([piece.value for piece in pieces], dtype=np.int32)

        observation = {
            'board': self._board,
            'pieces': pieces
        }

        return observation

    def _convert_to_keys(self, action: dict[str, int]):
        hold = Moves._holds[action['hold']]
        standard = Moves._standards[action['standard']]
        spin = Moves._spins[action['spin']]

        key_sequence = hold + standard + spin + [Keys.HARD_DROP]

        return key_sequence

    def _try_key_vector(self, key_vector: np.ndarray):
        # Key vector is delta [row, column, rotation]

        delta_loc = key_vector[:-1]
        delta_r = key_vector[-1]

        if delta_r:
            self._rotate(delta_r)
        else:
            self._move(delta_loc)

    def _rotate(self, try_delta_r: int):
        new_r = (self._active_piece.r + try_delta_r) % 4
        cells = self._rotation_system.orientations[self._active_piece.piece_type][new_r]

        if not self._rotation_system.overlaps(cells=cells, loc=self._active_piece.loc):
            # Applying rotation doesn't overlap, so do it
            self._active_piece.r = new_r
            self._active_piece.delta_r = try_delta_r

            self._active_piece.delta_loc = np.zeros((2,), dtype=np.int32)
            self._active_piece.cells = cells

        else:
            # Overlaps, so try kicks
            kick_table = (self._rotation_system.i_kicks
                          if self._active_piece.piece_type == PieceType.I
                          else self._rotation_system.kicks)
            self._rotation_system.kick_piece(kick_table, self._active_piece, cells, new_r, try_delta_r)

    def _move(self, try_delta_loc: np.ndarray):
        # Vertical movement
        if try_delta_loc[0]:
            # -1 = up (NEVER)
            # +1 = down
            direction = np.sign(try_delta_loc[0])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_y in range(0, try_delta_loc[0] + direction, direction):
                new_delta_loc = np.array([delta_y, 0], dtype=np.int32)
                if self._rotation_system.overlaps(cells=self._active_piece.cells, loc=self._active_piece.loc + new_delta_loc):
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
                if self._rotation_system.overlaps(cells=self._active_piece.cells, loc=self._active_piece.loc + new_delta_loc):
                    break
                else:
                    delta_loc = new_delta_loc

        else:
            # Delta would have to be [0, 0] to get here
            raise ValueError(f"Should never reach this! Delta: {try_delta_loc}")

        self._active_piece.loc += delta_loc
        self._active_piece.delta_loc = delta_loc
        self._active_piece.delta_r = 0

    def _try_hold(self):
        # Using `_can_hold` is unnecessary for this implementation
        # since only valid action sequences exist and none include pressing
        # the hold key twice. This is included in case of future changes.
        if self._can_hold:
            if self._hold_piece == PieceType.N:
                # No piece held, so this is the first one
                self._hold_piece = self._active_piece.piece_type
                self._active_piece = self._spawn_piece(self._queue.pop(0))
            else:
                self._active_piece, self._hold_piece = (self._spawn_piece(self._hold_piece),
                                                        self._active_piece.piece_type)
            self._can_hold = False

    def _lock_piece(self) -> bool:
        # DOES NOT MOVE PIECE TO THE BOTTOM
        # This method assumes the piece is already in a settled position.
        # Returns True if piece locked successfully, False if died
        cell_coords = self._active_piece.cells + self._active_piece.loc

        self._board[cell_coords[:, 0], cell_coords[:, 1]] = 1

        self._clear_lines()

        self._active_piece = self._spawn_piece(self._queue.pop(0))

        if np.any(self._board[:4] != 0):
            return False
        else:
            return True

    def _fill_queue(self) -> list[PieceType]:
        while len(self._queue) < self._queue_size:
            self._queue.append(self._next_bag.pop(0))
            if len(self._next_bag) == 0:
                self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))

    def _clear_lines(self):
        self._clears = 0
        for i, row in enumerate(self._board):
            if np.all(row != 0):
                self._clears += 1
                self._board[0] = 0
                self._board[1:i+1] = self._board[:i]
