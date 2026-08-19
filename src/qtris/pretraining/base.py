"""Shared pretraining helpers: checkpoint resolution + dataset loading."""

import math
import os

import tensorflow as tf

from qtris.config import PretrainConfig

_pretrain_cfg = PretrainConfig()


def resolve_resume_checkpoint(resume_from, manager):
    """Pick the checkpoint to restore from.

    `resume_from` (the `--resume-from` flag) may be a checkpoint directory (its
    latest is used) or a specific ckpt prefix; falls back to the manager's own
    latest when not given. New checkpoints always save to the manager's dir.
    """
    if resume_from:
        resume_from = str(resume_from)
        return tf.train.latest_checkpoint(resume_from) or resume_from
    return manager.latest_checkpoint


class PretrainerBase:
    """Dataset loading + value-label calibration. Subclasses provide
    `_train_step` and `train`."""

    def __init__(
        self,
        dataset_path,
        value_anchor_q=_pretrain_cfg.value_anchor_q,
        value_anchor_t=_pretrain_cfg.value_anchor_t,
    ):
        if not 0.0 < value_anchor_t < 1.0:
            raise ValueError(f"value_anchor_t must be in (0, 1), got {value_anchor_t}")
        if not 0.5 < value_anchor_q < 1.0:
            raise ValueError(
                f"value_anchor_q must be in (0.5, 1), got {value_anchor_q}"
            )
        self._dataset_path = dataset_path
        self._value_anchor_q = value_anchor_q
        self._value_anchor_t = value_anchor_t
        self._value_scale = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="value_scale"
        )
        self._value_center = tf.Variable(
            0.0, trainable=False, dtype=tf.float32, name="value_center"
        )
        self._value_var = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="value_var"
        )

    def _load_dataset_placement(self, batch_size):
        """Load the 128-slot placement dataset (cand_placements + cand_scores).

        Calibrates the bounded tanh value label; the policy target is built per batch
        in the train step from cand_scores."""
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(
                f"No dataset at {self._dataset_path}. Run `uv run datagen` to collect one."
            )

        dataset = tf.data.Dataset.load(self._dataset_path)
        spec = dataset.element_spec
        if "cand_placements" not in spec or "cand_scores" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} is not the placement schema (needs "
                "`cand_placements` + `cand_scores`). Regenerate with `uv run datagen`."
            )

        self._assign_tanh_value_norm(dataset)

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

    def _load_eval_placement(self, val_path, batch_size):
        """Load a SEPARATE, frozen held-out placement set for validation top1/top3.

        Must be a dataset the model NEVER trains on (collect it once to its own path,
        never merge it into the training dataset). An in-dataset split is NOT a valid
        generalization signal here: warm-started runs have already trained on every
        transition in the training file, so a carved-out 'val' subset is already
        memorized. This separate never-trained set is the only honest held-out metric."""
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"No val dataset at {val_path}.")
        ds = tf.data.Dataset.load(val_path)
        spec = ds.element_spec
        if "cand_placements" not in spec or "cand_scores" not in spec:
            raise ValueError(
                f"Val dataset at {val_path} is not the placement schema "
                "(needs `cand_placements` + `cand_scores`)."
            )
        return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def _dataset_vmax(dataset):
        """Per-position max over legal candidate scores (the oracle's best-move value)."""
        return tf.concat(
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

    def _assign_tanh_value_norm(self, dataset):
        """Calibrate the bounded value label `tanh((vmax - center) / scale)`.

        center is the median, so 0 means a typical board - what 0 also means to the 1v1
        AZ tanh head this warm-starts. scale places the anchor quantile at anchor_t,
        leaving range above it rather than saturating there. Both come from quantiles
        because the oracle score's upper tail runs ~30x the median (deep beam lines
        cashing a large surge)."""
        vmax = tf.sort(self._dataset_vmax(dataset))
        n = tf.shape(vmax)[0]

        def quantile(p):
            return vmax[tf.cast(tf.round(p * tf.cast(n - 1, tf.float32)), tf.int32)]

        center = quantile(0.5)
        span = quantile(self._value_anchor_q) - center
        scale = tf.maximum(span / math.atanh(self._value_anchor_t), 1e-3)
        target = tf.tanh((vmax - center) / scale)

        self._value_center.assign(center)
        self._value_scale.assign(scale)
        self._value_var.assign(tf.maximum(tf.math.reduce_variance(target), 1e-6))
        saturated = tf.reduce_mean(tf.cast(tf.abs(target) > 0.99, tf.float32))
        print(
            f"Value label | n={int(n)} | median={float(center):.2f} "
            f"q{100 * self._value_anchor_q:g}={float(center + span):.2f} | "
            f"scale={float(self._value_scale):.2f} | tanh target: "
            f"std={float(tf.sqrt(self._value_var)):.3f} "
            f"saturated={100.0 * float(saturated):.2f}%",
            flush=True,
        )

    def _tanh_value_target(self, vmax):
        """Apply the calibrated bounded label to a batch of per-position max scores."""
        return tf.tanh((vmax - self._value_center) / self._value_scale)
