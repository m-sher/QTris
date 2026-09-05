"""train_step's attack-head regression: masked rows carry no signal and attack_coef gates
the head's update."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from qtris.training.placement_az import train_step

from test_attack_head import C, _batch, _net


def _train_batch(n, target, mask):
    boards, pieces, bcg, pls, cand_mask = _batch(n, seed=1)
    pi = np.zeros((n, C), np.float32)
    pi[:, 0] = 1.0
    return {
        "boards": boards,
        "pieces": pieces,
        "bcg": bcg,
        "cand_placements": pls,
        "cand_mask": cand_mask,
        "pi_target": tf.constant(pi),
        "value_target": tf.constant(np.full(n, 0.3, np.float32)),
        "policy_mask": tf.ones(n, tf.float32),
        "attack_target": tf.constant(np.asarray(target, np.float32)),
        "attack_mask": tf.constant(np.asarray(mask, np.float32)),
    }


def _compiled():
    net = _net()
    net.compile(optimizer=keras.optimizers.Adam(1e-2))
    return net


def _attack_weights(net):
    return [w.numpy().copy() for w in net.attack_trunk.weights + net.attack_top.weights]


def _step(net, batch, attack_coef):
    """One eager train_step: the tf.function's body run directly."""
    return train_step.python_function(
        net, batch, tf.constant(1.0, tf.float32), tf.constant(attack_coef, tf.float32)
    )


def test_all_masked_rows_give_zero_loss_and_leave_the_head_still():
    net = _compiled()
    before = _attack_weights(net)
    out = _step(net, _train_batch(4, [1.0] * 4, [0.0] * 4), 1.0)
    assert float(out["attack_loss"]) == 0.0
    for a, b in zip(before, _attack_weights(net)):
        assert np.array_equal(a, b)


def test_masked_rows_do_not_enter_the_loss():
    """A fresh head reads 0 everywhere, so the loss over the two live rows is fixed
    whatever the masked rows' targets say."""
    out = _step(_compiled(), _train_batch(4, [0.9, 0.1, 5.0, -5.0], [1, 1, 0, 0]), 1.0)
    assert float(out["attack_loss"]) == pytest.approx(0.41, abs=1e-6)


def test_attack_coef_zero_freezes_the_head_but_not_the_value():
    net = _compiled()
    before = _attack_weights(net)
    value_before = net.value_top.weights[0].numpy().copy()
    _step(net, _train_batch(4, [0.9] * 4, [1.0] * 4), 0.0)
    for a, b in zip(before, _attack_weights(net)):
        assert np.array_equal(a, b)
    assert not np.array_equal(value_before, net.value_top.weights[0].numpy())


def test_attack_loss_falls_under_training():
    net = _compiled()
    batch = _train_batch(4, [0.9, 0.1, 0.9, 0.1], [1.0] * 4)
    losses = [float(_step(net, batch, 1.0)["attack_loss"]) for _ in range(20)]
    assert losses[-1] < losses[0]
