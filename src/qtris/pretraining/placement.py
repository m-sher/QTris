from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.pretraining.base import PretrainerBase
import tensorflow as tf
from tensorflow import keras


class Pretrainer(PretrainerBase):
    def __init__(
        self,
        dataset_path="datasets/tetris_expert_dataset_placement",
        policy_temp=10.0,
        value_weight=1.0,
    ):
        super().__init__(dataset_path)
        self._policy_temp = policy_temp
        self._value_weight = value_weight

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
        value_target = (
            tf.reduce_max(masked_scores, axis=-1, keepdims=True) / self._value_scale
        )  # (B, 1)

        with tf.GradientTape() as tape:
            logits, values = model(
                (board, pieces, bcg, cand_placements, mask), training=True
            )
            model_logp = tf.nn.log_softmax(logits, axis=-1)  # (B, C), illegal -> ~-inf
            policy_loss = tf.reduce_mean(-tf.reduce_sum(target * model_logp, axis=-1))
            value_loss = tf.reduce_mean(tf.square(values - value_target))
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

    def train(self, model, epochs=10, batch_size=256, checkpoint_manager=None):
        dataset = self._load_dataset_placement(batch_size=batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                policy_loss, top1, top3, value_loss = self._train_step(model, batch)
                if step % 100 == 0:
                    print(
                        f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                        f"Top1: {float(top1):1.3f} | Top3: {float(top3):1.3f} | "
                        f"Value: {float(value_loss):2.3f}",
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
    dropout_rate = 0.0
    batch_size = args.batch_size

    model = PlacementPolicyValueNet(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )
    optimizer = keras.optimizers.Adam(3e-4)
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
    }
    if args.dataset is not None:
        pretrainer_kwargs["dataset_path"] = str(args.dataset)
    pretrainer = Pretrainer(**pretrainer_kwargs)

    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, value_scale=pretrainer._value_scale
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "checkpoints/placement_pretrained_policy", max_to_keep=3
    )
    if checkpoint_manager.latest_checkpoint:
        # Resumes a merged checkpoint fully, or warm-starts the shared trunk +
        # policy head from an old policy-only checkpoint (value head stays fresh).
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(
            f"Restored checkpoint from {checkpoint_manager.latest_checkpoint}.",
            flush=True,
        )

    pretrainer.train(
        model,
        epochs=args.num_epochs,
        batch_size=batch_size,
        checkpoint_manager=checkpoint_manager,
    )
