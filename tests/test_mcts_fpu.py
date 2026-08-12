"""The FPU floor: what an unvisited child scores during PUCT selection."""

import numpy as np
import pytest
import tensorflow as tf

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.placement_features import MCTS_CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.search.cmcts import CMCTS
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS

SIMS = 96


def _make_env(seed=0):
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        garbage_chance=0.0,
        auto_push_garbage=False,
        auto_fill_queue=True,
        seed=seed,
        idx=0,
        num_row_tiers=2,
    )
    env._reset()
    return env


def _net(value_bias):
    """A tiny net whose value head is pinned to tanh(`value_bias`) for every input."""
    net = PlacementPolicyValueNet(
        batch_size=1,
        piece_dim=8,
        depth=16,
        num_heads=2,
        num_layers=1,
        dropout_rate=0.0,
        value_activation="tanh",
    )
    net(
        (
            tf.keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            tf.keras.Input(shape=(7,), dtype=tf.int64),
            tf.keras.Input(shape=(3,), dtype=tf.float32),
            tf.keras.Input(
                shape=(MCTS_CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            tf.keras.Input(shape=(MCTS_CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    top = net.value_top
    kernel, _bias = top.get_weights()
    top.set_weights([np.zeros_like(kernel), np.array([value_bias], np.float32)])
    return net


def _coverage(net, *, fpu_relative, fpu_reduction, seed=1):
    res = PlacementMCTS(
        net,
        MCTSConfig(
            num_simulations=SIMS,
            c_puct=1.5,
            dirichlet_eps=0.0,
            leaves_per_round=4,
            w_attack=0.0,
            w_death=1.0,
            w_b2b=0.0,
            gamma=1.0,
            fpu_relative=fpu_relative,
            fpu_reduction=fpu_reduction,
        ),
    ).search([_make_env(seed)], 1.0, np.zeros(1, np.float32))
    assert not res[0]["dead"]
    counts = res[0]["counts"]
    legal = int(res[0]["cand_mask"].sum())
    return int((counts > 0).sum()) / legal


def test_negative_q_forces_full_breadth_under_the_absolute_floor():
    """With every leaf valued below the 0 floor, an unvisited child outscores its visited
    siblings, so selection sweeps every legal candidate."""
    cov = _coverage(_net(-2.0), fpu_relative=0, fpu_reduction=0.0)
    assert cov == pytest.approx(1.0)


def test_relative_floor_narrows_the_sweep_at_negative_q():
    net = _net(-2.0)
    absolute = _coverage(net, fpu_relative=0, fpu_reduction=0.0)
    relative = _coverage(net, fpu_relative=1, fpu_reduction=0.25)
    assert relative < absolute
    assert relative < 0.7


def test_larger_reduction_is_never_wider():
    net = _net(-2.0)
    covs = [_coverage(net, fpu_relative=1, fpu_reduction=r) for r in (0.1, 0.5, 2.0)]
    assert covs == sorted(covs, reverse=True)


def test_absolute_floor_is_the_default():
    assert MCTSConfig().fpu_relative == 0
    assert MCTSConfig().fpu_reduction == 0.0
    engine = CMCTS(
        1, num_simulations=4, leaves_per_round=1, fpu_relative=1, fpu_reduction=0.3
    )
    engine.destroy()
