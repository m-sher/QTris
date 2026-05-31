"""Shared pretraining helpers (surge-correction return remapping).

Used by both `pretraining/ar.py` and `pretraining/flat.py` to normalize the
expert dataset's returns into the current env's reachable range. The math
below was a fix for a reward-shaping change in the env - see surge_correction
docstring.
"""

import os

import tensorflow as tf
from tensorflow import keras

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


class PretrainerBase:
    """Shared dataset loading for AR + Flat pretrainers.

    Subclasses (Pretrainer, FlatPretrainer) provide the family-specific
    `_train_step` and `train`; everything here is identical across both.
    """

    def __init__(self, dataset_path, policy_only=False):
        self._dataset_path = dataset_path
        self._policy_only = policy_only
        self._scc = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self._return_scale = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="return_scale"
        )
        self._value_scale = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="value_scale"
        )

    def _load_dataset(self, batch_size):
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(
                f"No dataset at {self._dataset_path}. Run `uv run datagen` to collect one."
            )

        dataset = tf.data.Dataset.load(self._dataset_path)
        spec = dataset.element_spec
        if "returns" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} lacks `returns` field. "
                "Regenerate with the current datagen (value pretraining requires returns)."
            )
        if "sample_weights" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} lacks `sample_weights` field. "
                "Regenerate with the current datagen."
            )

        if not self._policy_only:
            all_returns = tf.concat(
                [batch["returns"] for batch in dataset.batch(100_000)],
                axis=0,
            )
            all_b2b = tf.concat(
                [
                    batch["b2b_combo_garbage"][..., 0]
                    for batch in dataset.batch(100_000)
                ],
                axis=0,
            )
            corrected = all_returns + surge_correction(all_b2b)
            clipped = tf.clip_by_value(corrected, RETURN_CLIP_LOW, RETURN_CLIP_HIGH)

            n_total = tf.cast(tf.size(all_returns), tf.float32)
            n_clipped_low = tf.reduce_sum(
                tf.cast(corrected < RETURN_CLIP_LOW, tf.float32)
            )
            n_clipped_high = tf.reduce_sum(
                tf.cast(corrected > RETURN_CLIP_HIGH, tf.float32)
            )
            frac_clipped = (n_clipped_low + n_clipped_high) / n_total
            max_b2b = tf.reduce_max(all_b2b)

            clip_mean = tf.reduce_mean(clipped)
            clip_std = tf.math.reduce_std(clipped)
            scale = tf.maximum(clip_std, 1.0)
            self._return_scale.assign(scale)
            print(
                f"Returns | n={int(n_total)} | max_b2b in dataset={float(max_b2b):.0f} "
                f"| clipped to [{RETURN_CLIP_LOW:.0f}, {RETURN_CLIP_HIGH:.0f}]: "
                f"{float(n_clipped_low):.0f} low + {float(n_clipped_high):.0f} high "
                f"({100.0 * float(frac_clipped):.2f}%) "
                f"| post: mean={float(clip_mean):.3f} std={float(clip_std):.3f} "
                f"| value-head scale={float(scale):.3f}",
                flush=True,
            )

        cached = dataset.cache()
        for _ in cached:
            pass

        return (
            cached.shuffle(buffer_size=500_000)
            .batch(
                batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def _load_dataset_dense(self, batch_size):
        """Load the dense 320-action dataset (cand_sequences + cand_scores).

        Computes the value-head scale from each position's max legal score so the
        value target is standardized; the policy target is built per batch in the
        train step from cand_scores."""
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(
                f"No dataset at {self._dataset_path}. Run `uv run datagen` to collect one."
            )

        dataset = tf.data.Dataset.load(self._dataset_path)
        spec = dataset.element_spec
        if "cand_scores" not in spec or "cand_sequences" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} is not the dense schema (needs "
                "`cand_scores` + `cand_sequences`). Regenerate with `uv run datagen ar`."
            )

        if not self._policy_only:
            self._assign_value_scale(dataset)

        cached = dataset.cache()
        for _ in cached:
            pass

        return (
            cached.shuffle(buffer_size=500_000)
            .batch(
                batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def _load_dataset_placement(self, batch_size):
        """Load the 128-slot placement dataset (cand_placements + cand_scores).

        Same value-head standardization as the dense loader; the policy target is
        built per batch in the train step from cand_scores."""
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(
                f"No dataset at {self._dataset_path}. Run `uv run datagen` to collect one."
            )

        dataset = tf.data.Dataset.load(self._dataset_path)
        spec = dataset.element_spec
        if "cand_placements" not in spec or "cand_scores" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} is not the placement schema (needs "
                "`cand_placements` + `cand_scores`). Regenerate with `uv run datagen placement`."
            )

        if not self._policy_only:
            self._assign_value_scale(dataset)

        cached = dataset.cache()
        for _ in cached:
            pass

        return (
            cached.shuffle(buffer_size=500_000)
            .batch(
                batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def _assign_value_scale(self, dataset):
        """Standardize the value head from each position's max legal candidate score."""
        vmax = tf.concat(
            [
                tf.reduce_max(
                    tf.where(
                        batch["cand_scores"] > -1e29,
                        batch["cand_scores"],
                        tf.constant(-1e30, dtype=tf.float32),
                    ),
                    axis=-1,
                )
                for batch in dataset.batch(100_000)
            ],
            axis=0,
        )
        v_mean = tf.reduce_mean(vmax)
        v_std = tf.math.reduce_std(vmax)
        scale = tf.maximum(v_std, 1.0)
        self._value_scale.assign(scale)
        print(
            f"Value target | n={int(tf.size(vmax))} | mean={float(v_mean):.3f} "
            f"std={float(v_std):.3f} | value-head scale={float(scale):.3f}",
            flush=True,
        )

    @staticmethod
    def load_expert_dataset(path, batch_size):
        dataset = tf.data.Dataset.load(path)
        if "sample_weights" not in dataset.element_spec:

            def _add_default_weight(x):
                return {**x, "sample_weights": tf.constant(1.0, dtype=tf.float32)}

            dataset = dataset.map(_add_default_weight)
        cached = dataset.cache()
        for _ in cached:
            pass
        return (
            cached.repeat()
            .shuffle(buffer_size=100_000)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
