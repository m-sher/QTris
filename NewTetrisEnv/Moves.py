import numpy as np

class Keys:
    START = 0
    HOLD = 1
    TAP_LEFT = 2
    TAP_RIGHT = 3
    DAS_LEFT = 4
    DAS_RIGHT = 5
    CLOCKWISE = 6
    ANTICLOCKWISE = 7
    ROTATE_180 = 8
    SOFT_DROP = 9
    HARD_DROP = 10
    PAD = 11
    
    key_vectors = {
        # vertical, horizontal, rotation
        TAP_LEFT: np.array([+0, -1, +0], dtype=np.int32),
        TAP_RIGHT: np.array([+0, +1, +0], dtype=np.int32),
        DAS_LEFT: np.array([+0, -100, +0], dtype=np.int32),
        DAS_RIGHT: np.array([+0, +100, +0], dtype=np.int32),
        CLOCKWISE: np.array([+0, +0, +1], dtype=np.int32),
        ANTICLOCKWISE: np.array([+0, +0, -1], dtype=np.int32),
        ROTATE_180: np.array([+0, +0, +2], dtype=np.int32),
        SOFT_DROP: np.array([+100, +0, +0], dtype=np.int32), # soft and hard drop
        HARD_DROP: np.array([+100, +0, +0], dtype=np.int32), # are the same deltas
    }
    
class Moves:
    _holds = [
        [],
        [Keys.HOLD]
    ]
    _standards = [
        [Keys.DAS_LEFT],
        [Keys.DAS_LEFT, Keys.TAP_RIGHT],
        [Keys.TAP_LEFT, Keys.TAP_LEFT],
        [Keys.TAP_LEFT],
        [],
        [Keys.TAP_RIGHT],
        [Keys.TAP_RIGHT, Keys.TAP_RIGHT],
        [Keys.DAS_RIGHT, Keys.TAP_LEFT],
        [Keys.DAS_RIGHT],                                       # No rotations
        
        [Keys.DAS_LEFT, Keys.CLOCKWISE],
        [Keys.TAP_LEFT, Keys.TAP_LEFT, Keys.CLOCKWISE],
        [Keys.TAP_LEFT, Keys.CLOCKWISE],
        [Keys.CLOCKWISE],
        [Keys.TAP_RIGHT, Keys.CLOCKWISE],
        [Keys.TAP_RIGHT, Keys.TAP_RIGHT, Keys.CLOCKWISE],
        [Keys.DAS_RIGHT, Keys.TAP_LEFT, Keys.CLOCKWISE],
        [Keys.DAS_RIGHT, Keys.CLOCKWISE],                       # CLOCKWISE rotations
        
        [Keys.DAS_LEFT, Keys.ANTICLOCKWISE],
        [Keys.TAP_LEFT, Keys.TAP_LEFT, Keys.ANTICLOCKWISE],
        [Keys.TAP_LEFT, Keys.ANTICLOCKWISE],
        [Keys.ANTICLOCKWISE],
        [Keys.TAP_RIGHT, Keys.ANTICLOCKWISE],
        [Keys.TAP_RIGHT, Keys.TAP_RIGHT, Keys.ANTICLOCKWISE],
        [Keys.DAS_RIGHT, Keys.TAP_LEFT, Keys.ANTICLOCKWISE],
        [Keys.DAS_RIGHT, Keys.ANTICLOCKWISE],                   # ANTICLOCKWISE rotations
        
        [Keys.DAS_LEFT, Keys.ROTATE_180],
        [Keys.TAP_LEFT, Keys.TAP_LEFT, Keys.ROTATE_180],
        [Keys.TAP_LEFT, Keys.ROTATE_180],
        [Keys.ROTATE_180],
        [Keys.TAP_RIGHT, Keys.ROTATE_180],
        [Keys.TAP_RIGHT, Keys.TAP_RIGHT, Keys.ROTATE_180],
        [Keys.DAS_RIGHT, Keys.TAP_LEFT, Keys.ROTATE_180],
        [Keys.DAS_RIGHT, Keys.ROTATE_180],                      # 180 rotations
        
        [Keys.CLOCKWISE, Keys.DAS_LEFT],
        [Keys.ANTICLOCKWISE, Keys.DAS_RIGHT],                   # Spin first DAS
    ]
    _spins = [
        [],
        [Keys.SOFT_DROP, Keys.CLOCKWISE],
        [Keys.SOFT_DROP, Keys.ANTICLOCKWISE],
        [Keys.SOFT_DROP, Keys.ROTATE_180],
        [Keys.SOFT_DROP, Keys.CLOCKWISE, Keys.CLOCKWISE],
        [Keys.SOFT_DROP, Keys.ANTICLOCKWISE, Keys.ANTICLOCKWISE],
        [Keys.SOFT_DROP, Keys.CLOCKWISE, Keys.ROTATE_180],
        [Keys.SOFT_DROP, Keys.ANTICLOCKWISE, Keys.ROTATE_180]
    ]