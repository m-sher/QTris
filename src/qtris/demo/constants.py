"""Shared display constants for all Tetris demo renderers."""

import numpy as np

# Piece type -> RGB color mapping (index 0 = empty/black)
PIECE_COLORS = np.array(
    [
        [0, 0, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 127, 0],
        [255, 200, 0],
        [0, 255, 0],
        [255, 0, 255],
        [255, 0, 0],
        [127, 127, 127],
    ]
)

# Sidebar glyphs: (piece_id, 4, 5) binary masks for empty + I J L O S T Z.
# Indexed by piece type id; used by colorize_piece_sidebar.
PIECE_DISPLAY = np.array(
    [
        # 0 empty
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        # 1 I
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        # 2 J
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        # 3 L
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        # 4 O
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        # 5 S
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        # 6 T
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        # 7 Z
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

# Key ID -> short display string
READABLE_KEYS = {
    1: "h",
    2: "l",
    3: "r",
    4: "L",
    5: "R",
    6: "c",
    7: "a",
    8: "1",
    9: "s",
    10: "H",
}

# BCG attention heatmap colors (b2b=cyan, combo=yellow, garbage=red)
BCG_COLORS_RGB = np.array(
    [
        [100, 200, 255],
        [255, 255, 100],
        [255, 100, 100],
    ]
)
BCG_LABELS = ["B2B", "Combo", "Garb"]
