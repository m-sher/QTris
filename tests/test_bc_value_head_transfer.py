"""How a BC checkpoint's value head reaches each AZ trainer.

The placement pretrainer trains its value head with a tanh activation against a bounded
label. 1v1 AZ shares that geometry and takes the head; solo AZ's head is linear over an
unbounded return, so it must not.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.training._1v1_placement_az import warm_start_full
from qtris.training.placement_az import warm_start_policy_only


def _net(value_activation):
    """A tiny built PlacementPolicyValueNet."""
    net = PlacementPolicyValueNet(
        batch_size=2,
        piece_dim=8,
        depth=16,
        num_heads=2,
        num_layers=1,
        dropout_rate=0.0,
        value_activation=value_activation,
    )
    net(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(7,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    return net


@pytest.fixture(scope="module")
def bc_checkpoint(tmp_path_factory):
    """A BC-shaped checkpoint: tanh value head, weights perturbed off their init."""
    src = _net("tanh")
    for w in src.weights:
        w.assign(w + 0.05)
    path = tf.train.Checkpoint(model=src).save(
        str(tmp_path_factory.mktemp("bc") / "ckpt")
    )
    return path


def _value_weights(net):
    return [w.numpy().copy() for w in net.value_trunk.weights + net.value_top.weights]


def test_bc_checkpoint_carries_a_value_head(bc_checkpoint):
    names = [n for n, _ in tf.train.list_variables(bc_checkpoint)]
    assert any("value_top" in n for n in names)


def test_1v1_takes_the_value_head(bc_checkpoint):
    net = _net("tanh")
    before_value = _value_weights(net)
    before_policy = net.score_top.weights[0].numpy().copy()

    assert warm_start_full(net, bc_checkpoint) is True

    after_value = _value_weights(net)
    assert not any(np.array_equal(a, b) for a, b in zip(after_value, before_value))
    assert not np.array_equal(net.score_top.weights[0].numpy(), before_policy)


def test_solo_keeps_its_value_head_fresh(bc_checkpoint):
    net = _net(None)
    before_value = _value_weights(net)
    before_policy = net.score_top.weights[0].numpy().copy()

    warm_start_policy_only(net, bc_checkpoint)

    after_value = _value_weights(net)
    assert all(np.array_equal(a, b) for a, b in zip(after_value, before_value))
    # The policy/trunk still warm-starts; only the value head is held back.
    assert not np.array_equal(net.score_top.weights[0].numpy(), before_policy)


def test_policy_only_checkpoint_reports_no_value_head(tmp_path):
    """A checkpoint without a value head restores partial, so the return value has to
    report that rather than let the caller read a fresh head as pretrained."""
    src = _net("tanh")
    policy_only = tf.train.Checkpoint(
        score_top=src.score_top, move_encoder=src.move_encoder
    )
    path = policy_only.save(str(tmp_path / "ckpt"))
    assert warm_start_full(_net("tanh"), path) is False
