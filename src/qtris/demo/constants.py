"""Shared display constants for all Tetris demo renderers."""
import numpy as np

# Piece type -> RGB color mapping (index 0 = empty/black)
PIECE_COLORS = np.array([
    [0, 0, 0],
    [0, 255, 255],
    [0, 0, 255],
    [255, 127, 0],
    [255, 200, 0],
    [0, 255, 0],
    [255, 0, 255],
    [255, 0, 0],
    [127, 127, 127],
])

# Key ID -> short display string
READABLE_KEYS = {
    1: "h", 2: "l", 3: "r", 4: "L", 5: "R",
    6: "c", 7: "a", 8: "1", 9: "s", 10: "H",
}

# BCG attention heatmap colors (b2b=cyan, combo=yellow, garbage=red)
BCG_COLORS_RGB = np.array([
    [100, 200, 255],
    [255, 255, 100],
    [255, 100, 100],
])
BCG_LABELS = ["B2B", "Combo", "Garb"]
