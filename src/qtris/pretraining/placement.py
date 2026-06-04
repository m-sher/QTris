from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.pretraining.base import PretrainerBase, resolve_resume_checkpoint
import tensorflow as tf
from tensorflow import keras


class Pretrainer(PretrainerBase):
    def __init__(
        self,
        dataset_path="datasets/tetris_oracle_placement",
        policy_temp=1.0,
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
        train_ds = self._load_dataset_placement(batch_size=batch_size)
        val_ds = (
            self._load_eval_placement(val_dataset_path, batch_size)
            if val_dataset_path
            else None
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
    optimizer = keras.optimizers.AdamW(
        learning_rate=3e-4, weight_decay=getattr(args, "weight_decay", 0.0)
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
