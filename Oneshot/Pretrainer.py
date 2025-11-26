from TetrisModel import PolicyModel
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


class Pretrainer:
    def __init__(
        self,
        dataset_root: str | Path = "../autoregressive_expert_dataset",
        board_cols: int = 10,
    ):
        self._dataset_root = Path(dataset_root)
        self._board_cols = board_cols
        self._rotations = 4
        self._kick_states = 2
        self._per_hold = self._board_cols * self._rotations * self._kick_states
        self.scc = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def _list_shards(self) -> list[str]:
        if not tf.io.gfile.isdir(self._dataset_root.as_posix()):
            raise FileNotFoundError(
                f"Dataset root `{self._dataset_root}` not found. "
                "Run OneshotPretrainGen.py to generate expert shards."
            )
        shard_paths = sorted(
            [
                (self._dataset_root / shard_name).as_posix()
                for shard_name in tf.io.gfile.listdir(self._dataset_root.as_posix())
                if shard_name.startswith("shard_")
            ]
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No dataset shards found under `{self._dataset_root}`."
            )
        return shard_paths

    def _compose_action_index(
        self, hold: tf.Tensor, column: tf.Tensor, rotation: tf.Tensor, is_kick: tf.Tensor
    ) -> tf.Tensor:
        hold = tf.cast(hold, tf.int64)
        column = tf.cast(column, tf.int64)
        rotation = tf.cast(rotation, tf.int64)
        is_kick = tf.cast(is_kick, tf.int64)

        per_rotation = self._board_cols * self._kick_states
        return hold * self._per_hold + rotation * per_rotation + column * self._kick_states + is_kick

    def _prepare_example(self, example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        action = self._compose_action_index(
            example["hold"], example["column"], example["rotation"], example["is_kick"]
        )
        return {
            "boards": tf.cast(example["board"], tf.float32),
            "pieces": tf.cast(example["pieces"], tf.int64),
            "b2b_combo": tf.cast(example["b2b_combo"], tf.float32),
            "actions": action,
        }

    def _load_dataset(self, batch_size: int | None = 1024) -> tf.data.Dataset:
        """
        Load the autoregressive expert shards produced by OneshotPretrainGen.
        """

        shard_paths = self._list_shards()
        dataset = tf.data.Dataset.load(shard_paths[0])
        for shard_path in shard_paths[1:]:
            dataset = dataset.concatenate(tf.data.Dataset.load(shard_path))

        dataset = dataset.map(
            self._prepare_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        ).cache().shuffle(1_000_000)

        if batch_size:
            dataset = dataset.batch(
                batch_size,
                deterministic=False,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)

        return dataset

    @tf.function
    def _train_step(self, model: keras.Model, batch) -> tuple[float, float]:
        """
        Perform a single training step.
        """
        board = batch["boards"]
        piece_seq = batch["pieces"]
        b2b_combo = batch["b2b_combo"]
        target_action = batch["actions"]

        with tf.GradientTape() as tape:
            logits, _ = model(
                (board, piece_seq, b2b_combo),
                training=True,
                return_scores=True,
            )
            loss = self.scc(target_action, logits)
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Compute accuracy
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(logits, axis=-1, output_type=tf.int64), target_action
                ),
                tf.float32,
            )
        )

        return loss, accuracy

    def train(
        self,
        model: keras.Model,
        epochs: int = 10,
        batch_size: int = 1024,
        checkpoint_manager=None,
    ):
        """
        Train the model on saved dataset.
        """
        # Load dataset
        dataset = self._load_dataset(batch_size)

        # Train model
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                # Perform training step
                loss, accuracy = self._train_step(model, batch)
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(
                        f"Step {step + 1} | Loss: {loss:2.3f} | Accuracy: {accuracy:1.3f}",
                        flush=True,
                    )
            # Save checkpoint after each epoch
            if checkpoint_manager is not None:
                checkpoint_manager.save()


def main():
    # Model params
    piece_dim = 8
    depth = 64
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.1
    batch_size = 1024

    queue_size = 5
    action_dim = 160

    # Initialize model and optimizer
    model = PolicyModel(
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=action_dim,
    )

    optimizer = keras.optimizers.Adam(3e-4, clipnorm=0.5)
    model.compile(optimizer=optimizer, jit_compile=True)
    print("Initialized model and optimizer.", flush=True)

    model.build(input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 2)])

    model.summary()
    
    # Initialize checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "./pretrained_checkpoints", max_to_keep=3
    )
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("Initialized checkpoint manager.", flush=True)

    pretrainer = Pretrainer()
    pretrainer.train(
        model, batch_size=batch_size, epochs=10, checkpoint_manager=checkpoint_manager
    )


if __name__ == "__main__":
    main()
