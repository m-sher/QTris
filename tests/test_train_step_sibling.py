"""train_step's sibling rows: their own loss term, gated by sibling_coef, and kept out
of the played rows' value loss."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from qtris.training.placement_az import train_step

from test_attack_head import C, _batch, _net


def _rows(n, seed, value_target, sibling):
    boards, pieces, bcg, pls, cand_mask = _batch(n, seed=seed)
    pi = np.zeros((n, C), np.float32)
    pi[:, 0] = 1.0
    return {
        "boards": boards,
        "pieces": pieces,
        "bcg": bcg,
        "cand_placements": pls,
        "cand_mask": cand_mask,
        "pi_target": tf.constant(pi),
        "value_target": tf.constant(np.full(n, value_target, np.float32)),
        "policy_mask": tf.constant(np.full(n, 0.0 if sibling else 1.0, np.float32)),
        "attack_target": tf.zeros(n, tf.float32),
        "attack_mask": tf.zeros(n, tf.float32),
        "sibling_mask": tf.constant(np.full(n, 1.0 if sibling else 0.0, np.float32)),
    }


def _concat(a, b):
    return {
        k: tf.concat([tf.convert_to_tensor(a[k]), tf.convert_to_tensor(b[k])], 0)
        for k in a
    }


def _compiled():
    net = _net()
    net.compile(optimizer=keras.optimizers.Adam(1e-2))
    return net


def _values(net, rows):
    return net.state_value(
        tf.convert_to_tensor(rows["boards"]),
        tf.convert_to_tensor(rows["pieces"]),
        tf.convert_to_tensor(rows["bcg"]),
    ).numpy()[:, 0]


def _step(net, batch, sibling_coef):
    return train_step.python_function(
        net,
        batch,
        tf.constant(1.0, tf.float32),
        tf.constant(0.0, tf.float32),
        tf.constant(sibling_coef, tf.float32),
    )


def test_losses_split_by_the_sibling_mask():
    net = _compiled()
    played = _rows(6, 1, 0.3, sibling=False)
    sib = _rows(4, 2, -1.0, sibling=True)
    v_played, v_sib = _values(net, played), _values(net, sib)
    out = _step(net, _concat(played, sib), 1.0)
    assert float(out["value_loss"]) == pytest.approx(
        float(np.mean((v_played - 0.3) ** 2)), abs=1e-5
    )
    assert float(out["sibling_loss"]) == pytest.approx(
        float(np.mean((v_sib + 1.0) ** 2)), abs=1e-5
    )


def test_no_sibling_mask_means_no_sibling_term():
    net = _compiled()
    played = _rows(6, 1, 0.3, sibling=False)
    played.pop("sibling_mask")
    out = _step(net, played, 1.0)
    assert float(out["sibling_loss"]) == 0.0
    assert float(out["sibling_explained_var"]) == 0.0


def test_sibling_coef_gates_how_far_the_head_follows_sibling_targets():
    drops = []
    for coef in (0.0, 1.0):
        tf.random.set_seed(0)
        net = _compiled()
        played = _rows(6, 1, 0.3, sibling=False)
        sib = _rows(6, 2, -1.0, sibling=True)
        before = _values(net, sib).mean()
        for _ in range(8):
            _step(net, _concat(played, sib), coef)
        drops.append(before - _values(net, sib).mean())
    assert drops[1] > drops[0] + 0.05
