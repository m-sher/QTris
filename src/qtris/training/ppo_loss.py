"""Shared clipped-PPO math: surrogate loss term + clipped value loss.

Used by all 3 trainers (`training/{ar,flat,_1v1}.py`). The functions are
element-wise / reduced-scalar primitives — callers handle masking and
sequence-vs-scalar reduction themselves (the `_1v1.train_step_ar` variant
applies a per-token decision_mask before reducing).
"""
import tensorflow as tf


def clipped_surrogate(ratio, advantages, ppo_clip):
    """Element-wise clipped PPO surrogate.

    Returns ``(surrogate, clipped_ratio)`` where
    ``surrogate = min(ratio * advantages, clipped_ratio * advantages)`` and
    ``clipped_ratio = clip(ratio, 1 - ppo_clip, 1 + ppo_clip)``.

    Caller reduces (scalar variants: mean over batch; sequence variants:
    apply decision mask then mean-over-decisions then mean over batch).
    ``clipped_ratio`` is returned so the trainer can compute the
    ``clipped_frac`` metric without re-applying the clip.
    """
    clipped_ratio = tf.clip_by_value(ratio, 1 - ppo_clip, 1 + ppo_clip)
    surrogate = tf.minimum(ratio * advantages, clipped_ratio * advantages)
    return surrogate, clipped_ratio


def clipped_value_loss(values, old_values, returns, value_clip):
    """Standard PPO clipped value loss (Schulman et al. 2017).

    Scalar: ``mean(max(MSE(values, returns), MSE(clipped_values, returns)))``
    where ``clipped_values = old_values + clip(values - old_values, ±value_clip)``.
    """
    value_error = values - returns
    clipped_values = old_values + tf.clip_by_value(
        values - old_values, -value_clip, value_clip
    )
    clipped_value_error = clipped_values - returns
    return tf.reduce_mean(
        tf.maximum(tf.square(value_error), tf.square(clipped_value_error))
    )
