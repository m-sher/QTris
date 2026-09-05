"""The attack head: output shapes, the [0, 1] bound, its zero start, and its place at the
end of the variable list."""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from qtris.data.placement_features import MCTS_CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS

from test_mcts_rounds import _played_env

C = MCTS_CANDIDATE_CAPACITY


def _net():
    net = PlacementPolicyValueNet(
        batch_size=2,
        piece_dim=8,
        depth=16,
        num_heads=2,
        num_layers=1,
        dropout_rate=0.0,
        value_activation="tanh",
    )
    net(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(7,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(C, PLACEMENT_FEATURE_DIM), dtype=tf.float32),
            keras.Input(shape=(C,), dtype=tf.bool),
        )
    )
    return net


def _batch(n=3, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((n, C), bool)
    mask[:, :20] = True
    return (
        tf.constant(rng.random((n, 24, 10, 1), np.float32).round()),
        tf.constant(rng.integers(0, 7, (n, 7)).astype(np.int64)),
        tf.constant(rng.random((n, 3), np.float32)),
        tf.constant(rng.random((n, C, PLACEMENT_FEATURE_DIM), np.float32)),
        tf.constant(mask),
    )


def test_call_and_policy_value_return_three_heads():
    net = _net()
    b = _batch()
    logits, value, attack = net(b, training=False)
    assert tuple(logits.shape) == (3, C)
    assert tuple(value.shape) == (3, 1)
    assert tuple(attack.shape) == (3, 1)
    logits, value, attack = net.policy_value(b)
    assert tuple(logits.shape) == (3, C)
    assert tuple(value.shape) == (3, 1)
    assert tuple(attack.shape) == (3, 1)
    assert tuple(net.state_value(b[0], b[1], b[2]).shape) == (3, 1)


def test_fresh_attack_head_reads_zero_everywhere():
    _logits, _value, attack = _net().policy_value(_batch(seed=3))
    assert np.all(attack.numpy() == 0.0)


def test_attack_head_variables_come_last():
    net = _net()
    head = net.attack_trunk.trainable_variables + net.attack_top.trainable_variables
    assert len(head) == 6
    tail = net.trainable_variables[-len(head) :]
    assert [id(v) for v in tail] == [id(v) for v in head]


class _RampAttackNet:
    """Flat priors, zero value, and an attack head rising with the row index (or zero)."""

    def __init__(self, attack_on):
        self.attack_on = attack_on

    def policy_value(self, inputs):
        n = int(inputs[0].shape[0])
        logits = tf.zeros((n, C), tf.float32)
        if self.attack_on:
            attack = tf.reshape(tf.linspace(0.0, 1.0, n), (n, 1))
        else:
            attack = tf.zeros((n, 1), tf.float32)
        return logits, tf.zeros((n, 1), tf.float32), attack


def _search_counts(net, window):
    env = _played_env(7, 30)
    cfg = MCTSConfig(
        num_simulations=64,
        dirichlet_eps=0.0,
        leaves_per_round=8,
        gamma=1.0,
        w_attack=0.006,
        w_death=1.0,
        w_b2b=0.0,
        w_height=0.0,
        w_bumpiness=0.0,
        w_holes=0.0,
        w_plain=0.0,
        attack_window=window,
    )
    res = PlacementMCTS(net, cfg).search([env], 1.0, 0.0)[0]
    assert not res["dead"]
    return np.asarray(res["counts"], np.float64)


def test_attack_head_steers_the_search_only_when_on():
    off_zero = _search_counts(_RampAttackNet(False), 0)
    off_ramp = _search_counts(_RampAttackNet(True), 0)
    on_ramp = _search_counts(_RampAttackNet(True), 14)
    assert np.array_equal(off_zero, off_ramp)
    assert not np.array_equal(off_zero, on_ramp)
