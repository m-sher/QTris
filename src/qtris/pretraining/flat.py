from qtris.models.value import ValueModel
from qtris.models.flat.model import FlatPolicyModel
from qtris.pretraining.base import PretrainerBase, correct_and_clip
import tensorflow as tf
from tensorflow import keras


class FlatPretrainer(PretrainerBase):
    def __init__(
        self, dataset_path="datasets/tetris_expert_dataset_flat", policy_only=False
    ):
        super().__init__(dataset_path, policy_only)

    @tf.function
    def _train_step(self, p_model, v_model, batch):
        board = batch["boards"]
        pieces = batch["pieces"]
        bcg = batch["b2b_combo_garbage"]
        action_indices = batch["action_indices"]
        valid_masks = batch["valid_masks"]
        sample_weights = batch["sample_weights"]

        with tf.GradientTape() as p_tape:
            logits = p_model((board, pieces, bcg), training=True)
            masked_logits = tf.where(
                valid_masks, logits, tf.constant(-1e9, dtype=tf.float32)
            )
            per_sample_loss = self._scc(action_indices, masked_logits)
            policy_loss = tf.math.divide_no_nan(
                tf.reduce_sum(per_sample_loss * sample_weights),
                tf.reduce_sum(sample_weights),
            )

        p_gradients = p_tape.gradient(policy_loss, p_model.trainable_variables)
        p_model.optimizer.apply_gradients(zip(p_gradients, p_model.trainable_variables))

        predicted = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted, action_indices), tf.float32)
        )

        top3 = tf.math.in_top_k(action_indices, masked_logits, k=3)
        accuracy_top3 = tf.reduce_mean(tf.cast(top3, tf.float32))

        if self._policy_only:
            return (
                policy_loss,
                accuracy,
                accuracy_top3,
                tf.constant(0.0, dtype=tf.float32),
            )

        returns = correct_and_clip(batch["returns"], bcg[..., 0])
        with tf.GradientTape() as v_tape:
            values = v_model((board, pieces, bcg), training=True)
            targets = tf.reshape(returns / self._return_scale, (-1, 1))
            squared_error = tf.square(values - targets)
            value_loss = tf.math.divide_no_nan(
                tf.reduce_sum(squared_error * sample_weights[:, None]),
                tf.reduce_sum(sample_weights),
            )

        v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
        v_model.optimizer.apply_gradients(zip(v_gradients, v_model.trainable_variables))

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
                policy_loss, accuracy, accuracy_top3, value_loss = self._train_step(
                    p_model, v_model, batch
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


def main(args):
    piece_dim = 8
    depth = 64
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = args.batch_size
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
        p_checkpoint, "checkpoints/flat_pretrained_policy", max_to_keep=3
    )
    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained policy checkpoint.", flush=True)

    pretrainer_kwargs = {"policy_only": args.policy_only}
    if args.dataset is not None:
        pretrainer_kwargs["dataset_path"] = str(args.dataset)
    pretrainer = FlatPretrainer(**pretrainer_kwargs)

    v_checkpoint_manager = None
    if v_model is not None:
        v_checkpoint = tf.train.Checkpoint(
            model=v_model,
            optimizer=v_optimizer,
            return_scale=pretrainer._return_scale,
        )
        v_checkpoint_manager = tf.train.CheckpointManager(
            v_checkpoint, "checkpoints/flat_pretrained_value", max_to_keep=3
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
        epochs=args.num_epochs,
        batch_size=batch_size,
        p_checkpoint_manager=p_checkpoint_manager,
        v_checkpoint_manager=v_checkpoint_manager,
    )
