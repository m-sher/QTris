"""The placement pretrainer's bounded value label: calibration and transform.

Covers the properties the 1v1 AZ warm start depends on: every label is reachable by a
tanh head, 0 means a median board (AZ's neutral), and a heavy upper tail in the oracle
score leaves the rest of the distribution spread out.
"""

import numpy as np
import pytest
import tensorflow as tf

from qtris.pretraining.placement import Pretrainer

SENTINEL = -1e30
BULK, TAIL = 4000, 200


def _scores_dataset(vmax, n_cand=8, seed=0):
    """Placement-schema dataset whose per-position max legal score is exactly `vmax`."""
    rng = np.random.default_rng(seed)
    scores = np.full((len(vmax), n_cand), SENTINEL, dtype=np.float32)
    for i, v in enumerate(vmax):
        k = int(rng.integers(1, n_cand + 1))
        scores[i, :k] = v - rng.uniform(1.0, 50.0, size=k)
        scores[i, 0] = v
    return tf.data.Dataset.from_tensor_slices({"cand_scores": scores})


def _vmax(seed=0, tail=TAIL):
    """Oracle-score-shaped sample: a bulk plus a small, very heavy upper tail."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(100.0, 60.0, BULK)]
    if tail:
        parts.append(rng.uniform(2000.0, 9000.0, tail))
    return np.concatenate(parts).astype(np.float32)


def _calibrated(vmax, anchor_t=0.8, anchor_q=0.95):
    p = Pretrainer(
        dataset_path="unused", value_anchor_q=anchor_q, value_anchor_t=anchor_t
    )
    p._assign_tanh_value_norm(_scores_dataset(vmax))
    return p


def _label(p, values):
    return p._tanh_value_target(tf.constant(values, dtype=tf.float32)).numpy()


def test_center_is_the_median_and_reads_zero():
    v = _vmax()
    p = _calibrated(v)
    assert float(p._value_center) == pytest.approx(
        float(np.percentile(v, 50, method="nearest")), rel=1e-5
    )
    assert _label(p, [float(p._value_center)])[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("anchor_t", [0.5, 0.8, 0.9])
def test_anchor_quantile_reads_the_requested_level(anchor_t):
    v = _vmax()
    p = _calibrated(v, anchor_t=anchor_t)
    q95 = float(np.percentile(v, 95, method="nearest"))
    assert _label(p, [q95])[0] == pytest.approx(anchor_t, abs=1e-3)


def test_every_label_is_inside_the_tanh_head_range():
    """The tail keeps its place at the top of the ordering; it saturates together at 1.0
    in float32, so the top of the label range is a tie rather than a reshuffle."""
    v = _vmax()
    labels = _label(_calibrated(v), v)
    assert np.all(np.abs(labels) <= 1.0)
    assert labels[int(v.argmax())] == labels.max()
    assert labels[:BULK].max() < labels[BULK:].min()


def test_heavy_tail_does_not_crush_the_bulk():
    """With 5% of the sample running 30x the median, ordinary boards still spread over
    most of the head's range."""
    bulk = _vmax(tail=0)
    bulk_labels = _label(_calibrated(_vmax()), bulk)
    spread = np.percentile(bulk_labels, 95) - np.percentile(bulk_labels, 5)
    assert spread > 1.0


def test_death_floor_score_lands_at_the_label_floor():
    """evaluate_state returns -1e6 on instant death, well above the -1e29 illegal
    sentinel, so it reaches the label."""
    p = _calibrated(_vmax())
    assert _label(p, [-1e6])[0] == pytest.approx(-1.0, abs=1e-6)


def test_value_var_matches_the_realized_label_variance():
    v = _vmax()
    p = _calibrated(v)
    assert float(p._value_var) == pytest.approx(float(np.var(_label(p, v))), rel=1e-4)


@pytest.mark.parametrize(
    ("q", "t"), [(0.95, 0.0), (0.95, 1.0), (0.95, 1.5), (0.5, 0.8), (1.0, 0.8)]
)
def test_out_of_range_anchors_are_rejected(q, t):
    with pytest.raises(ValueError):
        Pretrainer(dataset_path="unused", value_anchor_q=q, value_anchor_t=t)
