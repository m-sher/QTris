import tetris
from tetris import BaseGame
from tetris.impl.scorer import Scorer
from tetris.board import Board
from tetris.types import MoveDelta
from tetris.types import MoveKind
from tetris.types import PieceType
from typing import Optional
import numpy as np

class TetrisEnv():
    def __init__(self, scorer):
        self._game = BaseGame(tetris.impl.presets.Tetrio, {'queue_size': 5})
        self._scorer = scorer
        self._game.scorer = self._scorer
        self._mov_dict = {
            'L': tetris.types.Move(kind=tetris.types.MoveKind.DRAG, y=-10, auto=True),
            'R': tetris.types.Move(kind=tetris.types.MoveKind.DRAG, y=10, auto=True),
            'l': tetris.types.Move(kind=tetris.types.MoveKind.DRAG, y=-1, auto=True),
            'r': tetris.types.Move(kind=tetris.types.MoveKind.DRAG, y=1, auto=True),
            '1': tetris.types.Move(kind=tetris.types.MoveKind.ROTATE, r=2, auto=True),
            'a': tetris.types.Move(kind=tetris.types.MoveKind.ROTATE, r=-1, auto=True),
            'c': tetris.types.Move(kind=tetris.types.MoveKind.ROTATE, r=1, auto=True),
            's': tetris.types.Move(kind=tetris.types.MoveKind.SOFT_DROP, x=100, auto=True),
            'h': tetris.types.Move(kind=tetris.types.MoveKind.SWAP, auto=True),
            'H': tetris.types.Move(kind=tetris.types.MoveKind.HARD_DROP, auto=True),
        }

        self._piece_dict = {
            0: 0, # NULL
            1: 1, # I
            2: 4, # J
            3: 3, # L
            4: 7, # O
            5: 6, # S
            6: 2, # T
            7: 5, # Z
        }
        
        self.reset()

    def reset(self):
        self._game.reset()
        self._game.scorer = self._scorer
        self._last_score = self._game.score
        board, pieces = self._get_data()
        self._current_time_step = (board, pieces, 0, False)
        return self._current_time_step

    def step(self, action):
        errored = False
        for key in action:
            try:
                self._press_key(key)
                errored = False
            except:
                errored = True
                break
        reward = self._game.score - self._last_score
        self._last_score = self._game.score
        board, pieces = self._get_data()
        terminated = self._game.lost or errored
        self._current_time_step = (board, pieces, reward, terminated)
        return self._current_time_step
   
    def current_time_step(self):
        return self._current_time_step
    
    def _press_key(self, key):
        self._game.push(self._mov_dict[key])
    
    def _get_data(self):
        board = self._get_board()
        pieces = self._get_pieces()
        return board, pieces
    
    def _get_board(self):
        board = (np.array(self._game.board[-28:]) != 0).astype(np.int32)
        return board
    
    def _get_pieces(self):
        active = self._game.piece.type.value
        hold = 0 if not self._game.hold else self._game.hold.value
        queue = [piece.value for piece in self._game.queue[:5]]
        return [self._piece_dict[piece] for piece in [active] + [hold] + queue]

class CustomScorer(Scorer):

    def __init__(self, score: Optional[int] = None, level: Optional[int] = None):
        self.score = score or 0
        self.level = level or 1
        self.line_clears = 0
        self.goal = self.level * 10
        self.combo = 0
        self.back_to_back = 0
        self.tspin = None
        self.tspin_mini = None

    def judge(self, delta: MoveDelta) -> None:  # noqa: D102
        piece = delta.game.piece
        board = delta.game.board

        # useful: https://tetris.wiki/T-Spin#Current_rules
        if delta.kind == MoveKind.ROTATE and piece.type == PieceType.T and delta.r != 0:
            
            px = piece.x
            py = piece.y
            corners = []
            # check corners clockwise from top-left
            for x, y in [(0, 0), (0, 2), (2, 2), (2, 0)]:
                corners.append(
                    x + px not in range(board.shape[0])
                    or y + py not in range(board.shape[1])
                    or board[x + px, y + py] != 0
                )

            back = None
            # find the back of the piece clockwise from top. note how this
            # is checked in the same order as the corners: corners[back] will
            # be the corner before the back edge (behind counter-clockwise)
            for i, pos in enumerate([(0, 1), (1, 2), (2, 1), (1, 0)]):
                if pos not in piece.minos:
                    back = i
                    break

            # ideally, an edge is always found. unless:
            #  - the piece's center is not 1,1 (should this be checked?)
            #  - its not T-shaped
            if back is not None:
                front_corners = corners[(back + 2) % 4] + corners[(back + 3) % 4]
                back_corners = corners[(back + 0) % 4] + corners[(back + 1) % 4]
                # if there are two corners in the front edge..
                if front_corners == 2 and back_corners >= 1:
                    # it's a proper t-spin!
                    self.tspin = True
                    self.tspin_mini = False
                # but, if there is one corner in front edge and two in the back..
                elif front_corners == 1 and back_corners == 2:
                    # this is still a tspin!
                    if abs(delta.x) == 2 and abs(delta.y) == 1:
                        # the piece was kicked far, proper t-spin!
                        self.tspin = True
                        self.tspin_mini = False
                    else:
                        # last case, mini-tspin
                        self.tspin = False
                        self.tspin_mini = True
                else:
                    self.tspin = False
                    self.tspin_mini = False
            else:
                self.tspin = False
                self.tspin_mini = False

        if delta.kind == MoveKind.SOFT_DROP:  # soft drop
            self.tspin = False
            self.tspin_mini = False
            if not delta.auto:
                self.score += delta.x

        elif delta.kind == MoveKind.HARD_DROP:  # hard drop
            score = 0

            if not delta.auto:  # not done by gravity
                self.score += delta.x * 2
                # self.score instead because hard drop isn't affected by level

            line_clears = len(delta.clears)  # how many lines cleared
            perfect_clear = all(not any(row) for row in board)
            
            if line_clears:  # B2B and combo
                if self.tspin or self.tspin_mini or line_clears >= 4 or perfect_clear:
                    self.back_to_back += 1
                else:
                    self.back_to_back = 0
                self.combo += 1
            else:
                self.combo = 0

            if perfect_clear:
                score += [0, 5, 6, 7, 9][line_clears]

            elif self.tspin:
                score += [0, 2, 4, 6, 0][line_clears]

            elif self.tspin_mini:
                score += [0, 0, 1, 2, 0][line_clears]

            else:
                score += [0, 0, 1, 2, 4][line_clears]

            # if self.combo:
            #     score += self.combo - 1

            # score *= self.level

            if self.back_to_back > 1:
                score += self.back_to_back - 1

            self.score += score
            self.line_clears += line_clears
            self.tspin = False
            self.tspin_mini = False

        elif delta.kind == MoveKind.DRAG:
            self.tspin = False
            self.tspin_mini = False