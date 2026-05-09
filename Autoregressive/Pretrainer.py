from TetrisEnv.Moves import Keys
from TetrisModel import PolicyModel, ValueModel
import argparse
import os
import tensorflow as tf
from tensorflow import keras


class Pretrainer:
    def __init__(self, dataset_path="../tetris_expert_dataset_b2b", policy_only=False):
        self._dataset_path = dataset_path
        self._policy_only = policy_only
        self._scc = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self._return_scale = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="return_scale"
        )

    def _load_dataset(self, batch_size):
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(
                f"No dataset at {self._dataset_path}. Run DataGen.py to collect one."
            )

        dataset = tf.data.Dataset.load(self._dataset_path)
        spec = dataset.element_spec
        if "returns" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} lacks `returns` field. "
                "Regenerate with the current DataGen.py (value pretraining requires returns)."
            )
        if "sample_weights" not in spec:
            raise ValueError(
                f"Dataset at {self._dataset_path} lacks `sample_weights` field. "
                "Regenerate with the current DataGen.py."
            )

        if not self._policy_only:
            all_returns = tf.concat(
                [batch["returns"] for batch in dataset.batch(100_000)],
                axis=0,
            )
            return_mean = tf.reduce_mean(all_returns)
            return_std = tf.math.reduce_std(all_returns)
            scale = tf.maximum(return_std, 1.0)
            self._return_scale.assign(scale)
            print(
                f"Return stats: mean={float(return_mean):.3f}, "
                f"std={float(return_std):.3f}, value-head scale={float(scale):.3f}",
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
        actions = batch["actions"]
        masks = batch["masks"]
        sample_weights = batch["sample_weights"]

        input_seq = actions[:, :-1]
        target_seq = actions[:, 1:]
        valid_mask = masks[:, 1:, :]

        pad_mask = tf.cast(target_seq != Keys.PAD, tf.float32)
        num_valid = tf.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1)
        decision_mask = tf.cast(num_valid > 1, tf.float32) * pad_mask
        weighted_mask = decision_mask * sample_weights[:, None]

        with tf.GradientTape() as p_tape:
            logits = p_model(
                (board, pieces, bcg, input_seq), training=True
            )
            masked_logits = tf.where(
                valid_mask, logits, tf.constant(-1e9, dtype=tf.float32)
            )
            per_token_loss = self._scc(target_seq, masked_logits)
            policy_loss = tf.math.divide_no_nan(
                tf.reduce_sum(per_token_loss * weighted_mask),
                tf.reduce_sum(weighted_mask),
            )

        p_gradients = p_tape.gradient(policy_loss, p_model.trainable_variables)
        p_model.optimizer.apply_gradients(
            zip(p_gradients, p_model.trainable_variables)
        )

        pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        correct = tf.cast(pred == target_seq, tf.float32) * decision_mask
        accuracy = tf.math.divide_no_nan(
            tf.reduce_sum(correct), tf.reduce_sum(decision_mask)
        )

        # Top-3 on decision tokens. Robust to "equivalent key sequence"
        # label noise — when two prefixes both reach the same final
        # placement (e.g. left-rotate vs rotate-left), DataGen records one
        # canonically and top-1 marks the other "wrong" even though it's
        # equally valid. Top-3 accepts the equivalent token in the top
        # candidates and tracks how often the right answer is *ranked*
        # well, which is closer to what placement-level decoding cares
        # about.
        last_dim = tf.shape(masked_logits)[-1]
        flat_logits = tf.reshape(masked_logits, [-1, last_dim])
        flat_targets = tf.reshape(target_seq, [-1])
        top3_flat = tf.math.in_top_k(flat_logits, flat_targets, k=3)
        top3 = tf.cast(
            tf.reshape(top3_flat, tf.shape(target_seq)), tf.float32
        ) * decision_mask
        accuracy_top3 = tf.math.divide_no_nan(
            tf.reduce_sum(top3), tf.reduce_sum(decision_mask)
        )

        if self._policy_only:
            return (
                policy_loss, accuracy, accuracy_top3,
                tf.constant(0.0, dtype=tf.float32),
            )

        returns = batch["returns"]
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
                "v_model is required unless Pretrainer was constructed "
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
        description="Pretrain the autoregressive policy (and optionally the "
                    "value head)."
    )
    ap.add_argument(
        "--policy-only",
        action="store_true",
        help="Train only the policy head; skip building, loading, and "
             "training the value model.",
    )
    args = ap.parse_args()

    piece_dim = 8
    key_dim = 12
    depth = 64
    max_len = 15
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = 512

    p_model = PolicyModel(
        batch_size=batch_size,
        piece_dim=piece_dim,
        key_dim=key_dim,
        depth=depth,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=key_dim,
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
            output_dim=1,
        )
        v_optimizer = keras.optimizers.Adam(3e-4)
        v_model.compile(optimizer=v_optimizer, jit_compile=True)
    print("Initialized models and optimizers.", flush=True)

    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(max_len,), dtype=tf.int64),
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
        p_checkpoint, "./pretrained_checkpoints", max_to_keep=3
    )
    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained policy checkpoint.", flush=True)

    pretrainer = Pretrainer(policy_only=args.policy_only)

    v_checkpoint_manager = None
    if v_model is not None:
        v_checkpoint = tf.train.Checkpoint(
            model=v_model,
            optimizer=v_optimizer,
            return_scale=pretrainer._return_scale,
        )
        v_checkpoint_manager = tf.train.CheckpointManager(
            v_checkpoint, "./pretrained_value_checkpoints", max_to_keep=3
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
