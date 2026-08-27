import numpy as np


def overlaps(cells: np.ndarray, loc: np.ndarray, board: np.ndarray) -> bool:
    cell_coords = cells + loc
    rows, cols = cell_coords.T

    # Outside board vertically
    if np.any(rows < 0) or np.any(rows > board.shape[0] - 1):
        return True
    # Outside board horizontally
    if np.any(cols < 0) or np.any(cols > board.shape[1] - 1):
        return True
    # Overlaps occupied cell
    if np.any(board[rows, cols] != 0):
        return True

    return False
