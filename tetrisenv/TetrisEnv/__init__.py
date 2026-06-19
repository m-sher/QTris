"""
TFTetrisEnv

TensorFlow environment for RL with Tetris SRS+.
"""

from .Pieces import PieceType, Piece
from .Moves import Keys, Moves
from .RotationSystem import RotationSystem
from .Scorer import Scorer
from .PyTetrisEnv import PyTetrisEnv, spawn_envelope_blocked
from .PyTetris1v1Env import PyTetris1v1Env
from .PyTetrisRunner import PyTetrisRunner
from .Py1v1TetrisRunner import Py1v1TetrisRunner
from .TetrioRandom import TetrioRNG
from .KeySequences import KeySequenceFinder

__version__ = "0.8.0"

__all__ = [
    "PieceType",
    "Piece",
    "Keys",
    "Moves",
    "RotationSystem",
    "Scorer",
    "PyTetrisEnv",
    "spawn_envelope_blocked",
    "PyTetris1v1Env",
    "PyTetrisRunner",
    "Py1v1TetrisRunner",
    "TetrioRNG",
    "KeySequenceFinder",
]


def info():
    """Return basic package information."""
    return f"TFTetrisEnv version {__version__}: TensorFlow environment for RL with Tetris SRS+."
