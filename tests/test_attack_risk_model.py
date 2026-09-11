import hashlib
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from qtris.search.cmcts import CANDIDATE_CAPACITY as CAP, RISK_HORIZON as H
from qtris.training._1v1_placement_az import _build_net
from qtris.training.attack_risk import (
    attack_risk_config,
    load_profile,
    prepare_destination,
    save_profile,
    train_step,
)
from qtris.training.placement_az import warm_start_policy_only


def _net(risk=True):
    return _build_net(2, 8, 16, 2, 1, 5, attack_risk=risk)


def _inputs():
    rng = np.random.default_rng(7)
    return (
        tf.constant(rng.integers(0, 2, (2, 24, 10, 1)), tf.float32),
        tf.ones((2, 7), tf.int64),
        tf.zeros((2, 3)),
        tf.constant(rng.normal(size=(2, CAP, 18)), tf.float32),
        tf.ones((2, CAP), tf.bool),
    )


def _batch():
    inputs = _inputs()
    pi = np.zeros((2, CAP), np.float32)
    pi[:, 0] = 1
    return dict(
        zip(("boards", "pieces", "bcg", "cand_placements", "cand_mask"), inputs),
        gate_mask=inputs[-1],
        pi_target=tf.constant(pi),
        value_target=tf.constant([0.3, 0.4]),
        policy_mask=tf.constant([1.0, 0.0]),
        slot=tf.constant([0, 1]),
        hazard_target=tf.ones((2, H)),
        hazard_mask=tf.constant([[1.0] + [0.0] * (H - 1)] * 2),
    )


def test_initial_heads_and_legacy_two_output_contract():
    net = _net()
    logits, values, risks = net.policy_attack_risk(_inputs())
    assert logits.shape == (2, CAP) and risks.shape == (2, CAP, H)
    assert np.all(values.numpy() == 0)
    np.testing.assert_allclose(risks.numpy()[..., -1], 0.05, atol=1e-6)
    assert np.all(np.diff(risks.numpy(), axis=-1) >= 0)
    old_logits, old_values = net.policy_value(_inputs())
    np.testing.assert_allclose(old_logits, logits, atol=1e-5)
    np.testing.assert_allclose(old_values, values, atol=1e-6)


@pytest.mark.parametrize("weights_only", [False, True])
def test_migration_retains_exact_policy_and_fresh_critics(tmp_path, weights_only):
    source = _net(False)
    source.compile(optimizer=tf.keras.optimizers.Adam(3e-4))
    source.optimizer.build(source.trainable_variables)
    for weight in source.optimizer.variables():
        weight.assign(tf.ones_like(weight))
    for weight in source.weights:
        weight.assign(weight + 0.02)
    prefix = str(tmp_path / "old")
    if weights_only:
        source.save_weights(prefix)
    else:
        tf.train.Checkpoint(model=source).write(prefix)
    hashes = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in tmp_path.iterdir()}
    target = _net()
    target.compile(optimizer=tf.keras.optimizers.Adam(3e-4))
    target.optimizer.build(target.trainable_variables)
    fresh_optimizer = [w.numpy().copy() for w in target.optimizer.variables()]
    critic_weights = (
        target.value_trunk.weights + target.value_top.weights + target.risk_top.weights
    )
    fresh = [w.numpy().copy() for w in critic_weights]
    warm_start_policy_only(target, prefix)
    a, _ = source(_inputs())
    b, _ = target(_inputs())
    np.testing.assert_allclose(a, b, atol=1e-6)
    for initial, current in zip(fresh, critic_weights):
        np.testing.assert_array_equal(initial, current)
    for initial, current in zip(fresh_optimizer, target.optimizer.variables()):
        np.testing.assert_array_equal(initial, current)
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_training_updates_both_critics_and_resume_restores_optimizer(tmp_path):
    net = _net()
    optimizer = tf.keras.optimizers.Adam(3e-4, clipnorm=0.5)
    net.compile(optimizer=optimizer)
    optimizer.build(net.trainable_variables)
    batch = _batch()
    first = train_step(net, batch, tf.constant(1.0), tf.constant(1.0))
    assert np.isfinite([float(v) for v in first.values()]).all()
    assert np.any(net.risk_top.kernel.numpy() != 0)
    assert np.any(net.value_top.kernel.numpy() != 0)
    assert optimizer.iterations.numpy() == 1
    prefix = tf.train.Checkpoint(model=net, optimizer=optimizer).save(
        str(tmp_path / "ckpt")
    )
    restored = _net()
    restored_opt = tf.keras.optimizers.Adam(3e-4, clipnorm=0.5)
    restored.compile(optimizer=restored_opt)
    restored_opt.build(restored.trainable_variables)
    tf.train.Checkpoint(model=restored, optimizer=restored_opt).restore(
        prefix
    ).assert_consumed()
    for a, b in zip(net.weights, restored.weights):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(optimizer.variables(), restored_opt.variables()):
        np.testing.assert_array_equal(a, b)


def test_censored_hazard_labels_cannot_change_loss():
    net = _net()
    net.compile(optimizer=tf.keras.optimizers.Adam(0.0))
    net.optimizer.build(net.trainable_variables)
    a = _batch()
    b = dict(a, hazard_target=tf.constant([[1.0] + [0.0] * (H - 1)] * 2))
    loss_a = train_step(net, a, tf.constant(1.0), tf.constant(1.0))["risk_loss"]
    loss_b = train_step(net, b, tf.constant(1.0), tf.constant(1.0))["risk_loss"]
    assert float(loss_a) == pytest.approx(float(loss_b))


def test_destination_profile_and_migration_rejection(tmp_path):
    cfg = attack_risk_config()
    source = tf.train.Checkpoint(model=_net(False)).save(
        str(tmp_path / "source" / "ckpt")
    )
    destination = tmp_path / "fresh"
    assert prepare_destination(destination, source, cfg, 14) == source
    profile = load_profile(destination)
    assert profile["risk_horizon"] == H
    assert profile["search"]["gamma"] == 0.97
    with pytest.raises(ValueError, match="empty destination"):
        prepare_destination(destination, source, cfg, 14)
    assert prepare_destination(destination, None, cfg, 14) is None
    with pytest.raises(ValueError, match="differ"):
        prepare_destination(destination, None, attack_risk_config(gamma=0.99), 14)
    with pytest.raises(ValueError, match="Legacy"):
        prepare_destination(Path(source).parent, None, cfg, 14)
    prefix = destination / "pool/gen_0"
    save_profile(str(prefix) + ".objective.json", cfg)
    assert load_profile(prefix) == profile


def test_demo_uses_each_checkpoint_profile(tmp_path, monkeypatch):
    from qtris.demo import placement_1v1 as demo

    monkeypatch.setattr(demo, "depth", 16)
    monkeypatch.setattr(demo, "num_heads", 2)
    monkeypatch.setattr(demo, "num_layers", 1)
    old = tf.train.Checkpoint(model=_net(False)).save(str(tmp_path / "old" / "ckpt"))
    new = tf.train.Checkpoint(model=_net()).save(str(tmp_path / "new" / "ckpt"))
    cfg = attack_risk_config(gamma=0.93, risk_threshold=0.15)
    save_profile(new + ".objective.json", cfg)
    assert demo.load_net(old).risk_horizon == 0
    assert demo.load_net(new).risk_horizon == H
    assert demo.load_search_config(old).gamma == 1
    assert demo.load_search_config(new).gamma == 0.93
    assert demo.load_search_config(new).risk_threshold == 0.15
    Path(new + ".objective.json").unlink()
    with pytest.raises(ValueError, match="disagree"):
        demo.load_net(new)


def test_checkpoint_sidecars_follow_manager_retention(tmp_path):
    from qtris.training.attack_risk import save_checkpoint

    checkpoint = tf.train.Checkpoint(value=tf.Variable(1.0))
    manager = tf.train.CheckpointManager(checkpoint, str(tmp_path), max_to_keep=1)
    first = save_checkpoint(manager, attack_risk_config(), 14)
    second = save_checkpoint(manager, attack_risk_config(), 14)
    assert not Path(first + ".objective.json").exists()
    assert Path(second + ".objective.json").exists()
