from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyModel
from qtris.models.value import ValueModel
from qtris.pretraining.base import PretrainerBase
import tensorflow as tf
from tensorflow import keras


class Pretrainer(PretrainerBase):
    def __init__(
        self,
        dataset_path="datasets/tetris_expert_dataset_placement",
        policy_only=False,
        policy_temp=10.0,
    ):
        super().__init__(dataset_path, policy_only)
        self._policy_temp = policy_temp

    @tf.function
    def _train_step(self, p_model, v_model, batch):
        board = batch["boards"]
        pieces = batch["pieces"]
        bcg = batch["b2b_combo_garbage"]
        cand_placements = batch["cand_placements"]  # (B, C, F)
        cand_scores = batch["cand_scores"]  # (B, C) f32, sentinel = illegal

        # Policy target: softmax(score/temp) over legal candidates; illegal -> 0.
        mask = cand_scores > -1e29  # (B, C)
        masked_scores = tf.where(mask, cand_scores, tf.constant(-1e30, tf.float32))
        target = tf.nn.softmax(masked_scores / self._policy_temp, axis=-1)  # (B, C)

        with tf.GradientTape() as p_tape:
            logits = p_model((board, pieces, bcg, cand_placements, mask), training=True)
            model_logp = tf.nn.log_softmax(logits, axis=-1)  # (B, C), illegal -> ~-inf
            policy_loss = tf.reduce_mean(-tf.reduce_sum(target * model_logp, axis=-1))

        p_gradients = p_tape.gradient(policy_loss, p_model.trainable_variables)
        p_model.optimizer.apply_gradients(zip(p_gradients, p_model.trainable_variables))

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

        if self._policy_only:
            return policy_loss, top1_agree, in_top3, tf.constant(0.0, dtype=tf.float32)

        value_target = (
            tf.reduce_max(masked_scores, axis=-1, keepdims=True) / self._value_scale
        )
        with tf.GradientTape() as v_tape:
            values = v_model((board, pieces, bcg), training=True)
            value_loss = tf.reduce_mean(tf.square(values - value_target))

        v_gradients = v_tape.gradient(value_loss, v_model.trainable_variables)
        v_model.optimizer.apply_gradients(zip(v_gradients, v_model.trainable_variables))

        return policy_loss, top1_agree, in_top3, value_loss

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

        dataset = self._load_dataset_placement(batch_size=batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                policy_loss, top1, top3, value_loss = self._train_step(
                    p_model, v_model, batch
                )
                if step % 100 == 0:
                    if self._policy_only:
                        print(
                            f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                            f"Top1: {float(top1):1.3f} | Top3: {float(top3):1.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                            f"Top1: {float(top1):1.3f} | Top3: {float(top3):1.3f} | "
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

    p_model = PlacementPolicyModel(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
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
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
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
        p_checkpoint, "checkpoints/placement_pretrained_policy", max_to_keep=3
    )
    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained policy checkpoint.", flush=True)

    pretrainer_kwargs = {
        "policy_only": args.policy_only,
        "policy_temp": args.policy_temp,
    }
    if args.dataset is not None:
        pretrainer_kwargs["dataset_path"] = str(args.dataset)
    pretrainer = Pretrainer(**pretrainer_kwargs)

    v_checkpoint_manager = None
    if v_model is not None:
        v_checkpoint = tf.train.Checkpoint(
            model=v_model,
            optimizer=v_optimizer,
            value_scale=pretrainer._value_scale,
        )
        v_checkpoint_manager = tf.train.CheckpointManager(
            v_checkpoint, "checkpoints/placement_pretrained_value", max_to_keep=3
        )
        if v_checkpoint_manager.latest_checkpoint:
            v_checkpoint.restore(
                v_checkpoint_manager.latest_checkpoint
            ).expect_partial()
            print(
                f"Restored pretrained value checkpoint "
                f"(value_scale={float(pretrainer._value_scale):.3f}).",
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
