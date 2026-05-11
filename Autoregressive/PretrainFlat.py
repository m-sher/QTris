from TetrisModelFlat import FlatPolicyModel, ValueModel
import argparse
import os
import tensorflow as tf
from tensorflow import keras


RETURN_CLIP_LOW = -150.0
RETURN_CLIP_HIGH = 100.0


class FlatPretrainer:
    def __init__(self, dataset_path="../tetris_expert_dataset_flat", policy_only=False):
        self._dataset_path = dataset_path
        self._policy_only = policy_only
        self._scc = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self._return_scale = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="return_scale"
        )

    @staticmethod
    def _surge_correction(b2b):
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

    @staticmethod
    def _correct_and_clip(returns, b2b):
        """Apply surge correction then clip to new-env reachable range."""
        corrected = returns + FlatPretrainer._surge_correction(b2b)
        return tf.clip_by_value(corrected, RETURN_CLIP_LOW, RETURN_CLIP_HIGH)

    def _load_dataset(self, batch_size):
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(
                f"No dataset at {self._dataset_path}. Run DataGenFlat.py to collect one."
            )

        dataset = tf.data.Dataset.load(self._dataset_path)
        spec = dataset.element_spec
        if "returns" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} lacks `returns` field. "
                "Regenerate with the current DataGenFlat.py (value pretraining requires returns)."
            )
        if "sample_weights" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} lacks `sample_weights` field. "
                "Regenerate with the current DataGenFlat.py."
            )

        if not self._policy_only:
            all_returns = tf.concat(
                [batch["returns"] for batch in dataset.batch(100_000)],
                axis=0,
            )
            all_b2b = tf.concat(
                [batch["b2b_combo_garbage"][..., 0] for batch in dataset.batch(100_000)],
                axis=0,
            )
            corrected = all_returns + self._surge_correction(all_b2b)
            clipped = tf.clip_by_value(corrected, RETURN_CLIP_LOW, RETURN_CLIP_HIGH)

            n_total = tf.cast(tf.size(all_returns), tf.float32)
            n_clipped_low = tf.reduce_sum(tf.cast(corrected < RETURN_CLIP_LOW, tf.float32))
            n_clipped_high = tf.reduce_sum(tf.cast(corrected > RETURN_CLIP_HIGH, tf.float32))
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
            cached
            .shuffle(buffer_size=500_000)
            .batch(
                batch_size,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .prefetch(tf.data.AUTOTUNE)
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
            cached
            .repeat()
            .shuffle(buffer_size=100_000)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    @tf.function
    def _train_step(self, p_model, v_model, batch):
        board = batch["boards"]
        pieces = batch["pieces"]
        bcg = batch["b2b_combo_garbage"]
        action_indices = batch["action_indices"]
        valid_masks = batch["valid_masks"]
        sample_weights = batch["sample_weights"]

        with tf.GradientTape() as p_tape:
            logits = p_model(
                (board, pieces, bcg), training=True
            )
            masked_logits = tf.where(
                valid_masks, logits, tf.constant(-1e9, dtype=tf.float32)
            )
            per_sample_loss = self._scc(action_indices, masked_logits)
            policy_loss = tf.math.divide_no_nan(
                tf.reduce_sum(per_sample_loss * sample_weights),
                tf.reduce_sum(sample_weights),
            )

        p_gradients = p_tape.gradient(policy_loss, p_model.trainable_variables)
        p_model.optimizer.apply_gradients(
            zip(p_gradients, p_model.trainable_variables)
        )

        predicted = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted, action_indices), tf.float32)
        )

        top3 = tf.math.in_top_k(action_indices, masked_logits, k=3)
        accuracy_top3 = tf.reduce_mean(tf.cast(top3, tf.float32))

        if self._policy_only:
            return (
                policy_loss, accuracy, accuracy_top3,
                tf.constant(0.0, dtype=tf.float32),
            )

        returns = self._correct_and_clip(batch["returns"], bcg[..., 0])
        with tf.GradientTape() as v_tape:
            values = v_model((board, pieces, bcg), training=True)
            targets = tf.reshape(returns / self._return_scale, (-1, 1))
            squared_error = tf.square(values - targets)
            value_loss = tf.math.divide_no_nan(
                tf.reduce_sum(squared_error * sample_weights[:, None]),
                tf.reduce_sum(sample_weights),
            )

        v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
        v_model.optimizer.apply_gradients(
            zip(v_gradients, v_model.trainable_variables)
        )

        return policy_loss, accuracy, accuracy_top3, value_loss

    def train(
        self,
        p_model,
        v_model=None,
        epochs=10,
        batch_size=256,
        p_checkpoint_manager=None,
        v_checkpoint_manager=None,
    ):
        if not self._policy_only and v_model is None:
            raise ValueError(
                "v_model is required unless FlatPretrainer was constructed "
                "with policy_only=True."
            )

        dataset = self._load_dataset(batch_size=batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                policy_loss, accuracy, accuracy_top3, value_loss = (
                    self._train_step(p_model, v_model, batch)
                )
                if step % 100 == 0:
                    if self._policy_only:
                        print(
                            f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                            f"Acc: {float(accuracy):1.3f} | "
                            f"Acc@3: {float(accuracy_top3):1.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                            f"Acc: {float(accuracy):1.3f} | "
                            f"Acc@3: {float(accuracy_top3):1.3f} | "
                            f"Value: {float(value_loss):2.3f}",
                            flush=True,
                        )
            if p_checkpoint_manager is not None:
                p_checkpoint_manager.save()
            if v_checkpoint_manager is not None and not self._policy_only:
                v_checkpoint_manager.save()


def main():
    ap = argparse.ArgumentParser(
        description="Pretrain the flat policy (and optionally the value head)."
    )
    ap.add_argument(
        "--policy-only",
        action="store_true",
        help="Train only the policy head; skip building, loading, and "
             "training the value model.",
    )
    args = ap.parse_args()

    piece_dim = 8
    depth = 64
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = 256
    num_row_tiers = 2
    num_sequences = 160 * num_row_tiers

    p_model = FlatPolicyModel(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        num_sequences=num_sequences,
    )

    p_optimizer = keras.optimizers.Adam(3e-4)
    p_model.compile(optimizer=p_optimizer, jit_compile=True)

    v_model = None
    v_optimizer = None
    if not args.policy_only:
        v_model = ValueModel(
            piece_dim=piece_dim,
            depth=depth,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            output_dim=1
        )
        v_optimizer = keras.optimizers.Adam(3e-4)
        v_model.compile(optimizer=v_optimizer, jit_compile=True)
    print("Initialized models and optimizers.", flush=True)

    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    p_model.summary()
    if v_model is not None:
        v_model(
            (
                keras.Input(shape=(24, 10, 1), dtype=tf.float32),
                keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
                keras.Input(shape=(3,), dtype=tf.float32),
            )
        )
        v_model.summary()

    p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint, "./pretrained_flat_policy_checkpoints", max_to_keep=3
    )
    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained policy checkpoint.", flush=True)

    pretrainer = FlatPretrainer(policy_only=args.policy_only)

    v_checkpoint_manager = None
    if v_model is not None:
        v_checkpoint = tf.train.Checkpoint(
            model=v_model,
            optimizer=v_optimizer,
            return_scale=pretrainer._return_scale,
        )
        v_checkpoint_manager = tf.train.CheckpointManager(
            v_checkpoint, "./pretrained_flat_value_checkpoints", max_to_keep=3
        )
        if v_checkpoint_manager.latest_checkpoint:
            v_checkpoint.restore(
                v_checkpoint_manager.latest_checkpoint
            ).expect_partial()
            print(
                f"Restored pretrained value checkpoint "
                f"(return_scale={float(pretrainer._return_scale):.3f}).",
                flush=True,
            )

    pretrainer.train(
        p_model,
        v_model,
        epochs=10,
        batch_size=batch_size,
        p_checkpoint_manager=p_checkpoint_manager,
        v_checkpoint_manager=v_checkpoint_manager,
    )


if __name__ == "__main__":
    main()
