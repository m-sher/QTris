"""Shared pretraining helpers (surge-correction return remapping).

Used by both `pretraining/ar.py` and `pretraining/flat.py` to normalize the
expert dataset's returns into the current env's reachable range. The math
below was a fix for a reward-shaping change in the env — see surge_correction
docstring.
"""

import tensorflow as tf

RETURN_CLIP_LOW = -150.0
RETURN_CLIP_HIGH = 100.0


def surge_correction(b2b):
    """Add back the recoverable part of the removed surge potential.

    Old env had ``φ`` with a ``surge_coef * (1.15^b2b - 1)`` term;
    new env removes it. Potential-shaping telescoping gives:

        G_new − G_old ≈ surge(b_{t-1}) − γ^{T-t+1} · surge(b_T)

    We can recover the first term exactly (b2b at the predicted
    state is in the dataset). The second term requires per-trajectory
    metadata we don't store, and explodes for trajectories where the
    beam expert chained b2b to extreme magnitudes (e.g. b2b > 100,
    where 1.15^b2b reaches 10^6+). The residual is bounded
    downstream by clipping the corrected returns to the range
    the new env can actually produce.
    """
    b2b = tf.cast(b2b, tf.float32)
    surge_lines = tf.where(b2b >= 4.0, b2b, tf.zeros_like(b2b))
    return tf.pow(1.15, surge_lines) - 1.0


def correct_and_clip(returns, b2b):
    """Apply surge correction then clip to new-env reachable range."""
    corrected = returns + surge_correction(b2b)
    return tf.clip_by_value(corrected, RETURN_CLIP_LOW, RETURN_CLIP_HIGH)
