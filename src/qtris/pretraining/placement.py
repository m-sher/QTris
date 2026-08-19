import math
import os

import tensorflow as tf
from tensorflow import keras

from qtris.config import PretrainConfig
from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet

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


class Pretrainer:
    """Behavioral cloning from the oracle's placement dataset: soft cross-entropy to
    the per-candidate scores, plus a bounded tanh value label."""

    def __init__(
        self,
        dataset_path="datasets/tetris_oracle_placement",
        policy_temp=1.0,
        value_weight=1.0,
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
        self._policy_temp = policy_temp
        self._value_weight = value_weight
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

    def _load_dataset(self, batch_size):
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

    def _load_eval(self, val_path, batch_size):
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

    @tf.function
    def _train_step(self, model, batch):
        board = batch["boards"]
        pieces = batch["pieces"]
        bcg = batch["b2b_combo_garbage"]
        cand_placements = batch["cand_placements"]  # (B, C, F)
        cand_scores = batch["cand_scores"]  # (B, C) f32, sentinel = illegal

        mask = cand_scores > -1e29  # (B, C)
        masked_scores = tf.where(mask, cand_scores, tf.constant(-1e30, tf.float32))
        target = tf.nn.softmax(masked_scores / self._policy_temp, axis=-1)  # (B, C)
        value_target = self._tanh_value_target(
            tf.reduce_max(masked_scores, axis=-1, keepdims=True)
        )  # (B, 1) in [-1, 1], 0 = median board

        with tf.GradientTape() as tape:
            logits, values = model(
                (board, pieces, bcg, cand_placements, mask), training=True
            )
            model_logp = tf.nn.log_softmax(logits, axis=-1)  # (B, C), illegal -> ~-inf
            policy_loss = tf.reduce_mean(-tf.reduce_sum(target * model_logp, axis=-1))
            # Fraction of variance unexplained (1.0 = predicting the mean), which keeps
            # value_weight scale-free.
            value_loss = (
                tf.reduce_mean(tf.square(values - value_target)) / self._value_var
            )
            loss = policy_loss + self._value_weight * value_loss

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Diagnostics: does the model's top candidate match the oracle's best, and
        # is the oracle's best within the model's top-3.
        best_slot = tf.argmax(target, axis=-1, output_type=tf.int32)  # (B,)
        top1_agree = tf.reduce_mean(
            tf.cast(
                tf.argmax(logits, -1, output_type=tf.int32) == best_slot, tf.float32
            )
        )
        top3 = tf.math.top_k(logits, k=3).indices  # (B, 3)
        in_top3 = tf.reduce_mean(
            tf.cast(tf.reduce_any(top3 == best_slot[:, None], axis=-1), tf.float32)
        )
        return policy_loss, top1_agree, in_top3, value_loss

    @tf.function
    def _eval_step(self, model, batch):
        """Held-out top1/top3 (no grad). Same target as `_train_step`, training=False.
        Returns summed hits + count so the caller can average over the val set."""
        cand_scores = batch["cand_scores"]
        mask = cand_scores > -1e29
        masked_scores = tf.where(mask, cand_scores, tf.constant(-1e30, tf.float32))
        target = tf.nn.softmax(masked_scores / self._policy_temp, axis=-1)

        logits, _ = model(
            (
                batch["boards"],
                batch["pieces"],
                batch["b2b_combo_garbage"],
                batch["cand_placements"],
                mask,
            ),
            training=False,
        )
        best_slot = tf.argmax(target, axis=-1, output_type=tf.int32)
        top1 = tf.reduce_sum(
            tf.cast(
                tf.argmax(logits, -1, output_type=tf.int32) == best_slot, tf.float32
            )
        )
        top3 = tf.math.top_k(logits, k=3).indices
        in_top3 = tf.reduce_sum(
            tf.cast(tf.reduce_any(top3 == best_slot[:, None], axis=-1), tf.float32)
        )
        n = tf.cast(tf.shape(best_slot)[0], tf.float32)
        return top1, in_top3, n

    def train(
        self,
        model,
        epochs=10,
        batch_size=256,
        checkpoint_manager=None,
        val_dataset_path=None,
    ):
        train_ds = self._load_dataset(batch_size=batch_size)
        val_ds = (
            self._load_eval(val_dataset_path, batch_size) if val_dataset_path else None
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(train_ds):
                policy_loss, top1, top3, value_loss = self._train_step(model, batch)
                if step % 100 == 0:
                    print(
                        f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                        f"Top1: {float(top1):1.3f} | Top3: {float(top3):1.3f} | "
                        f"Value: {float(value_loss):2.3f}",
                        flush=True,
                    )

            # Held-out validation on a SEPARATE never-trained set - the only honest
            # generalization signal (train Top1 is memorized-train accuracy).
            if val_ds is not None:
                v_top1 = v_top3 = v_n = 0.0
                for vbatch in val_ds:
                    t1, t3, n = self._eval_step(model, vbatch)
                    v_top1 += float(t1)
                    v_top3 += float(t3)
                    v_n += float(n)
                if v_n > 0:
                    print(
                        f"  val | Top1: {v_top1 / v_n:1.3f} | Top3: {v_top3 / v_n:1.3f} "
                        f"(held-out n={int(v_n)})",
                        flush=True,
                    )
            if checkpoint_manager is not None:
                checkpoint_manager.save()


def main(args):
    piece_dim = 8
    depth = 64
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.2
    batch_size = args.batch_size

    model = PlacementPolicyValueNet(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        value_activation="tanh",  # bounded label, matching the 1v1 AZ head
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=PretrainConfig().learning_rate,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    model.compile(optimizer=optimizer, jit_compile=True)
    model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    model.summary()
    print("Initialized model and optimizer.", flush=True)

    pretrainer_kwargs = {
        "policy_temp": args.policy_temp,
        "value_weight": args.value_weight,
        "value_anchor_t": args.value_anchor,
    }
    if args.dataset is not None:
        pretrainer_kwargs["dataset_path"] = str(args.dataset)
    pretrainer = Pretrainer(**pretrainer_kwargs)

    # Recalibrated from the dataset on every run, so restoring them has no effect; they
    # ride along so a consumer can read what the head was calibrated against.
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        value_scale=pretrainer._value_scale,
        value_center=pretrainer._value_center,
        value_var=pretrainer._value_var,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "checkpoints/placement_pretrained_policy", max_to_keep=3
    )
    resume = resolve_resume_checkpoint(
        getattr(args, "resume_from", None), checkpoint_manager
    )
    if resume:
        # Resumes a merged checkpoint fully, or warm-starts the shared trunk +
        # policy head from an old policy-only checkpoint (value head stays fresh).
        checkpoint.restore(resume).expect_partial()
        print(f"Restored checkpoint from {resume}.", flush=True)

    pretrainer.train(
        model,
        epochs=args.num_epochs,
        batch_size=batch_size,
        checkpoint_manager=checkpoint_manager,
        val_dataset_path=(
            str(args.val_dataset) if getattr(args, "val_dataset", None) else None
        ),
    )
