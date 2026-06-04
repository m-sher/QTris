from TetrisEnv.Moves import Keys
from qtris.models.ar.model import PolicyModel
from qtris.models.value import ValueModel
from qtris.pretraining.base import PretrainerBase, resolve_resume_checkpoint
import tensorflow as tf
from tensorflow import keras


class Pretrainer(PretrainerBase):
    def __init__(
        self,
        dataset_path="datasets/tetris_expert_dataset_b2b",
        policy_only=False,
        cand_topk=32,
        policy_temp=10.0,
        max_len=15,
    ):
        super().__init__(dataset_path, policy_only)
        self._cand_topk = cand_topk
        self._policy_temp = policy_temp
        self._max_len = max_len

    @tf.function
    def _train_step(self, p_model, v_model, batch):
        board = batch["boards"]
        pieces = batch["pieces"]
        bcg = batch["b2b_combo_garbage"]
        cand_seqs = batch["cand_sequences"]  # (B, A, L) int8
        cand_scores = batch["cand_scores"]  # (B, A) f32, sentinel = illegal

        B = tf.shape(board)[0]
        K = self._cand_topk
        L = self._max_len

        # Top-K candidate moves by score, with the search's best at index 0. The
        # target weight per candidate is softmax(score/temp); illegal slots get 0.
        topk_scores, topk_idx = tf.math.top_k(cand_scores, k=K)  # (B,K) desc
        legal = topk_scores > -1e29
        seqs_k = tf.cast(tf.gather(cand_seqs, topk_idx, batch_dims=1), tf.int64)
        masked_scores = tf.where(legal, topk_scores, tf.constant(-1e30, tf.float32))
        target = tf.nn.softmax(masked_scores / self._policy_temp, axis=-1)  # (B,K)

        with tf.GradientTape() as p_tape:
            # Encoder runs ONCE per position; only the cheap key decoder pays the
            # K multiplier (tile the context, score all K sequences as one batch).
            piece_dec, _ = p_model.process_obs((board, pieces, bcg), training=True)
            ctx = tf.repeat(piece_dec, K, axis=0)  # (B*K, C, D)
            s = tf.reshape(seqs_k, (B * K, L))
            logits, _ = p_model.process_keys((ctx, s[:, :-1]), training=True)
            logp = tf.nn.log_softmax(logits, axis=-1)  # (B*K, L-1, V)
            tok_logp = tf.gather(logp, s[:, 1:, None], batch_dims=2)[..., 0]
            pad = tf.cast(s[:, 1:] != Keys.PAD, tf.float32)
            seq_logp = tf.reshape(tf.reduce_sum(tok_logp * pad, axis=-1), (B, K))
            # Joint sequence distillation: maximize the target-weighted log-
            # likelihood of the candidate sequences (per-token weighted CE). This
            # trains the per-token policy that greedy decoding uses - a softmax
            # over per-sequence log-probs would only fix their relative ranking,
            # leaving per-token probs underdetermined - and has no length bias
            # against longer (spin) sequences.
            policy_loss = tf.reduce_mean(-tf.reduce_sum(target * seq_logp, axis=-1))

        p_gradients = p_tape.gradient(policy_loss, p_model.trainable_variables)
        p_model.optimizer.apply_gradients(zip(p_gradients, p_model.trainable_variables))

        # Diagnostics. `Gen`: greedy per-token accuracy along the best-scored
        # candidate - the behavior greedy generation produces. `Rank1`: whether
        # the model assigns that candidate the highest sequence log-prob.
        vocab = tf.shape(logits)[-1]
        best_logits = tf.reshape(logits, (B, K, L - 1, vocab))[:, 0]  # (B, L-1, V)
        best_tgt = seqs_k[:, 0, 1:]  # (B, L-1)
        gen_mask = tf.cast(best_tgt != Keys.PAD, tf.float32)
        gen_pred = tf.argmax(best_logits, axis=-1, output_type=tf.int64)
        accuracy = tf.math.divide_no_nan(
            tf.reduce_sum(tf.cast(gen_pred == best_tgt, tf.float32) * gen_mask),
            tf.reduce_sum(gen_mask),
        )
        accuracy_top3 = tf.reduce_mean(
            tf.cast(tf.argmax(seq_logp, axis=-1) == 0, tf.float32)
        )

        if self._policy_only:
            return (
                policy_loss,
                accuracy,
                accuracy_top3,
                tf.constant(0.0, dtype=tf.float32),
            )

        value_target = topk_scores[:, :1] / self._value_scale  # (B,1) = max score
        with tf.GradientTape() as v_tape:
            values = v_model((board, pieces, bcg), training=True)
            value_loss = tf.reduce_mean(tf.square(values - value_target))

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
                "v_model is required unless Pretrainer was constructed "
                "with policy_only=True."
            )

        dataset = self._load_dataset_dense(batch_size=batch_size)

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
                            f"Gen: {float(accuracy):1.3f} | "
                            f"Rank1: {float(accuracy_top3):1.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Step {step + 1} | Policy: {float(policy_loss):2.3f} | "
                            f"Gen: {float(accuracy):1.3f} | "
                            f"Rank1: {float(accuracy_top3):1.3f} | "
                            f"Value: {float(value_loss):2.3f}",
                            flush=True,
                        )
            if p_checkpoint_manager is not None:
                p_checkpoint_manager.save()
            if v_checkpoint_manager is not None and not self._policy_only:
                v_checkpoint_manager.save()


def main(args):
    piece_dim = 8
    key_dim = 12
    depth = 64
    max_len = 15
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = args.batch_size

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
        p_checkpoint, "checkpoints/ar_pretrained_policy", max_to_keep=3
    )
    p_resume = resolve_resume_checkpoint(
        getattr(args, "resume_from", None), p_checkpoint_manager
    )
    if p_resume:
        p_checkpoint.restore(p_resume).expect_partial()
        print(f"Restored pretrained policy checkpoint from {p_resume}.", flush=True)

    pretrainer_kwargs = {
        "policy_only": args.policy_only,
        "cand_topk": args.cand_topk,
        "policy_temp": args.policy_temp,
        "max_len": max_len,
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
            v_checkpoint, "checkpoints/ar_pretrained_value", max_to_keep=3
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
