from TetrisModelFlat import FlatPolicyModel, ValueModel
import os
import tensorflow as tf
from tensorflow import keras


class FlatPretrainer:
    def __init__(self, dataset_path="../tetris_expert_dataset_flat"):
        self._dataset_path = dataset_path
        self._scc = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

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
        returns = batch["returns"]

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

        with tf.GradientTape() as v_tape:
            values = v_model((board, pieces, bcg), training=True)
            targets = tf.reshape(returns, (-1, 1))
            squared_error = tf.square(values - targets)
            value_loss = tf.math.divide_no_nan(
                tf.reduce_sum(squared_error * sample_weights[:, None]),
                tf.reduce_sum(sample_weights),
            )

        v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
        v_model.optimizer.apply_gradients(
            zip(v_gradients, v_model.trainable_variables)
        )

        return policy_loss, accuracy, value_loss

    def train(
        self,
        p_model,
        v_model,
        epochs=10,
        batch_size=256,
        p_checkpoint_manager=None,
        v_checkpoint_manager=None,
    ):
        dataset = self._load_dataset(batch_size=batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                policy_loss, accuracy, value_loss = self._train_step(
                    p_model, v_model, batch
                )
                if step % 100 == 0:
                    print(
                        f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                        f"Acc: {float(accuracy):1.3f} | "
                        f"Value: {float(value_loss):2.3f}",
                        flush=True,
                    )
            if p_checkpoint_manager is not None:
                p_checkpoint_manager.save()
            if v_checkpoint_manager is not None:
                v_checkpoint_manager.save()


def main():
    piece_dim = 8
    depth = 64
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = 512
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

    v_model = ValueModel(
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=1,
    )

    p_optimizer = keras.optimizers.Adam(3e-4)
    p_model.compile(optimizer=p_optimizer, jit_compile=True)

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
    v_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    p_model.summary()
    v_model.summary()

    p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint, "./pretrained_flat_checkpoints", max_to_keep=3
    )
    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained policy checkpoint.", flush=True)

    v_checkpoint = tf.train.Checkpoint(model=v_model, optimizer=v_optimizer)
    v_checkpoint_manager = tf.train.CheckpointManager(
        v_checkpoint, "./pretrained_flat_value_checkpoints", max_to_keep=3
    )
    if v_checkpoint_manager.latest_checkpoint:
        v_checkpoint.restore(v_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained value checkpoint.", flush=True)

    pretrainer = FlatPretrainer()

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
