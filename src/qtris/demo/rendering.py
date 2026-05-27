"""Shared rendering helpers for Tetris demos.

Functions here produce numpy arrays for display; callers blit them to
pygame surfaces. No pygame dependency in this module.
"""
import numpy as np
import tensorflow as tf

from qtris.demo.constants import BCG_COLORS_RGB


def compute_bcg_heatmaps(piece_scores):
    """Build 3 colored (12, 5, 3) heatmaps for b2b, combo, garbage attention."""
    attention = tf.reduce_sum(piece_scores, axis=[0, 2])  # (batch, query, key)
    bcg_patch_attn = attention[0, 7:10, :60].numpy()  # (3, 60)
    grids = bcg_patch_attn.reshape(3, 12, 5)
    out = np.zeros((3, 12, 5, 3), dtype=np.uint8)
    for i in range(3):
        g = grids[i]
        g_min, g_max = g.min(), g.max()
        norm = (g - g_min) / (g_max - g_min + 1e-8)
        out[i] = (BCG_COLORS_RGB[i][None, None] * norm[..., None]).astype(np.uint8)
    return out


def draw_garbage_bar(env_instance, height=24, width=4):
    """Create a garbage queue visualization array (height, width, 3)."""
    surface = np.zeros((height, width, 3), dtype=np.uint8)
    current_row = height - 1
    for i, (num_rows, _, _timing) in enumerate(env_instance._garbage_queue):
        start_row = max(0, current_row - num_rows + 1)
        for row in range(start_row, current_row + 1):
            if 0 <= row < height:
                surface[row, :] = [255, 0, 0]
        if i < len(env_instance._garbage_queue) - 1 and start_row > 0:
            current_row = start_row - 2
        else:
            current_row = start_row - 1
        if current_row < 0:
            break
    return surface


def colorize_piece_sidebar(piece_display, pieces_array, piece_colors):
    """Color the 7-piece sidebar (28, 5, 3) using piece type colors.

    piece_display: (7, 4, 5) binary array from PieceDisplay.npy
    pieces_array: (7,) int array of piece type IDs
    piece_colors: (N, 3) color palette indexed by piece type
    """
    sidebar = piece_display[pieces_array].reshape((28, 5))
    type_colors = piece_colors[pieces_array]
    colored = np.zeros((28, 5, 3), dtype=np.uint8)
    for i in range(7):
        for r in range(4 * i, 4 * i + 4):
            for c in range(5):
                colored[r, c] = (type_colors[i] * sidebar[r, c]).astype(np.uint8)
    return colored
