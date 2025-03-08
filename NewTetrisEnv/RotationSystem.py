from .Pieces import PieceType, Piece
import numpy as np

class RotationSystem():
    
    def __init__(self, board):
        
        self._board = board
        
        self.orientations = {
            PieceType.I: np.array([
                [[1, 0], [1, 1], [1, 2], [1, 3]],
                [[0, 2], [1, 2], [2, 2], [3, 2]],
                [[2, 0], [2, 1], [2, 2], [2, 3]],
                [[0, 1], [1, 1], [2, 1], [3, 1]],
            ], dtype=np.int32),
            PieceType.J: np.array([
                [[0, 0], [1, 0], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [2, 1]],
                [[1, 0], [1, 1], [1, 2], [2, 2]],
                [[0, 1], [1, 1], [2, 0], [2, 1]],
            ], dtype=np.int32),
            PieceType.L: np.array([
                [[0, 2], [1, 0], [1, 1], [1, 2]],
                [[0, 1], [1, 1], [2, 1], [2, 2]],
                [[1, 0], [1, 1], [1, 2], [2, 0]],
                [[0, 0], [0, 1], [1, 1], [2, 1]],
            ], dtype=np.int32),
            PieceType.O: np.array([
                [[0, 1], [0, 2], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [1, 2]],
            ], dtype=np.int32),
            PieceType.S: np.array([
                [[0, 1], [0, 2], [1, 0], [1, 1]],
                [[0, 1], [1, 1], [1, 2], [2, 2]],
                [[1, 1], [1, 2], [2, 0], [2, 1]],
                [[0, 0], [1, 0], [1, 1], [2, 1]],
            ], dtype=np.int32),
            PieceType.T: np.array([
                [[0, 1], [1, 0], [1, 1], [1, 2]],
                [[0, 1], [1, 1], [1, 2], [2, 1]],
                [[1, 0], [1, 1], [1, 2], [2, 1]],
                [[0, 1], [1, 0], [1, 1], [2, 1]],
            ], dtype=np.int32),
            PieceType.Z: np.array([
                [[0, 0], [0, 1], [1, 1], [1, 2]],
                [[0, 2], [1, 1], [1, 2], [2, 1]],
                [[1, 0], [1, 1], [2, 1], [2, 2]],
                [[0, 1], [1, 0], [1, 1], [2, 0]],
            ], dtype=np.int32)            
        }
        
        self.kicks = {
            # (kick_from, kick_to): (offset1, offset2, offset3, offset4)
            (0, 1): np.array([[+0, -1], [-1, -1], [+2, +0], [+2, -1]], dtype=np.int32),  # 0 -> R | CW
            (0, 3): np.array([[+0, +1], [-1, +1], [+2, +0], [+2, +1]], dtype=np.int32),  # 0 -> L | CCW
            (1, 0): np.array([[+0, +1], [+1, +1], [-2, +0], [-2, +1]], dtype=np.int32),  # R -> 0 | CCW
            (1, 2): np.array([[+0, +1], [+1, +1], [-2, +0], [-2, +1]], dtype=np.int32),  # R -> 2 | CW
            (2, 1): np.array([[+0, -1], [-1, -1], [+2, +0], [+2, -1]], dtype=np.int32),  # 2 -> R | CCW
            (2, 3): np.array([[+0, +1], [-1, +1], [+2, +0], [+2, +1]], dtype=np.int32),  # 2 -> L | CW
            (3, 0): np.array([[+0, -1], [+1, -1], [-2, +0], [-2, -1]], dtype=np.int32),  # L -> 0 | CW
            (3, 2): np.array([[+0, -1], [+1, -1], [-2, +0], [-2, -1]], dtype=np.int32),  # L -> 2 | CCW
            (0, 2): np.array([[-1, +0], [-1, +1], [-1, -1], [+0, +1], [+0, -1]], dtype=np.int32),  # 0 -> 2
            (1, 3): np.array([[+0, +1], [-2, +1], [-1, +1], [-2, +0], [-1, +0]], dtype=np.int32),  # R -> L
            (2, 0): np.array([[+1, +0], [+1, -1], [+1, +1], [+0, -1], [+0, +1]], dtype=np.int32),  # 2 -> 0
            (3, 1): np.array([[+0, -1], [-2, -1], [-1, -1], [-2, +0], [-1, +0]], dtype=np.int32),  # L -> R
        }

        self.i_kicks = {
            # (kick_from, kick_to): (offset1, offset2, offset3, offset4)
            (0, 1): np.array([[+0, +1], [+0, -2], [+1, -2], [-2, +1]], dtype=np.int32),  # 0 -> R | CW
            (0, 3): np.array([[+0, -1], [+0, +2], [+1, +2], [-2, -1]], dtype=np.int32),  # 0 -> L | CCW
            (1, 0): np.array([[+0, -1], [+0, +2], [+2, -1], [-1, +2]], dtype=np.int32),  # R -> 0 | CCW
            (1, 2): np.array([[+0, -1], [+0, +2], [-2, -1], [+1, +2]], dtype=np.int32),  # R -> 2 | CW
            (2, 1): np.array([[+0, -2], [+0, +1], [-1, -2], [+2, +1]], dtype=np.int32),  # 2 -> R | CCW
            (2, 3): np.array([[+0, +2], [+0, -1], [-1, +2], [+2, -1]], dtype=np.int32),  # 2 -> L | CW
            (3, 0): np.array([[+0, +1], [+0, -2], [+2, +1], [-1, -2]], dtype=np.int32),  # L -> 0 | CW
            (3, 2): np.array([[+0, +1], [+0, -2], [-2, +1], [+1, -2]], dtype=np.int32),  # L -> 2 | CCW
            (0, 2): np.array([[-1, +0], [-1, +1], [-1, -1], [+0, +1], [+0, -1]], dtype=np.int32),  # 0 -> 2
            (1, 3): np.array([[+0, +1], [-2, +1], [-1, +1], [-2, +0], [-1, +0]], dtype=np.int32),  # R -> L
            (2, 0): np.array([[+1, +0], [+1, -1], [+1, +1], [+0, -1], [+0, +1]], dtype=np.int32),  # 2 -> 0
            (3, 1): np.array([[+0, -1], [-2, -1], [-1, -1], [-2, +0], [-1, +0]], dtype=np.int32),  # L -> R
        }
    
    def overlaps(self, cells: np.ndarray, loc: np.ndarray) -> bool:
        cell_coords = cells + loc
        rows, cols = cell_coords.T
        
        # Outside board vertically
        if np.any(rows < 0) or np.any(rows > self._board.shape[0] - 1):
            return True
        # Outside board horizontally
        if np.any(cols < 0) or np.any(cols > self._board.shape[1] - 1):
            return True
        # Overlaps occupied cell
        if np.any(self._board[rows, cols] != 0):
            return True
        
        return False
        
    def kick_piece(self, kick_table: dict[tuple[int, int], np.ndarray], piece: Piece, cells: np.ndarray, new_r: int, delta_r: int):
        if (piece.r, new_r) not in kick_table.keys():
            # Rotation not possible, do nothing
            piece.delta_r = 0
            piece.delta_loc = np.zeros((2,), dtype=np.int32)
            return
        
        for delta_loc in kick_table[(piece.r, new_r)]:
            # Check each kick and perform the first valid
            if not self.overlaps(cells=cells, loc=piece.loc + delta_loc):
                # Kick doesn't overlap, so apply it and break
                piece.r = new_r
                piece.delta_r = delta_r    
            
                piece.loc += delta_loc
                piece.delta_loc = delta_loc
                
                piece.cells = cells
                return 
    
        piece.delta_loc = np.zeros((2,), dtype=np.int32)
        piece.delta_r = 0