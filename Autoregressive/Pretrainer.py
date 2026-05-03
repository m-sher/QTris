from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from TetrisModel import PolicyModel, ValueModel
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras


def _build_mask(sequence, valid_sequences, max_len, key_dim):
    masks = np.zeros((max_len, key_dim), dtype=bool)
    for pos in range(1, max_len):
        prefix = sequence[:pos]
        match = np.all(valid_sequences[:, :pos] == prefix, axis=-1)
        if not match.any():
            continue
        next_tokens = valid_sequences[match, pos]
        masks[pos, np.unique(next_tokens)] = True
    return masks


def _collect(
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    key_dim,
    max_height,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    death_trim_count,
    gamma,
    log_every=1000,
):
    """Run a single env for num_steps total transitions, resetting on death.

    Per-episode discounted returns are computed in flush so that kept
    transitions carry the discounted impact of the upcoming death penalty
    even though the trimmed final tail isn't added to the dataset itself.
    """
    env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=max_steps_env,
        max_len=max_len,
        pathfinding=True,
        seed=seed,
        idx=0,
        garbage_chance=garbage_chance,
        garbage_min=garbage_min,
        garbage_max=garbage_max,
        garbage_push_delay=garbage_push_delay,
        auto_push_garbage=True,
        auto_fill_queue=True,
        num_row_tiers=num_row_tiers,
        gamma=gamma,
    )

    time_step = env.reset()
    searcher = CB2BSearch()

    transitions = []
    episode_buf = []
    unmatched = 0
    deaths = 0
    max_b2b = 0

    def flush(buf, is_death):
        if not buf:
            return
        returns_arr = np.zeros(len(buf), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(buf))):
            r = buf[t][6]
            d = float(buf[t][7])
            last = r + gamma * last * (1.0 - d)
            returns_arr[t] = last

        if is_death:
            kept_count = len(buf) - death_trim_count if len(buf) > death_trim_count else 0
        else:
            kept_count = len(buf)

        for t in range(kept_count):
            board, pieces, bcg, sequence, mask, sample_weight, _r, _d = buf[t]
            transitions.append(
                (board, pieces, bcg, sequence, mask, sample_weight, returns_arr[t])
            )

    for step in range(num_steps):
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)
        valid_sequences = obs["sequences"].astype(np.int64)

        action_idx, sequence = searcher.search(
            board=env._board,
            active_piece=env._active_piece.piece_type.value,
            hold_piece=env._hold_piece.value,
            queue=np.array(
                [p.value for p in env._queue], dtype=np.int32
            ),
            b2b=int(env._scorer._b2b),
            combo=int(env._scorer._combo),
            total_garbage=int(env._get_total_garbage()),
            garbage_push_delay=env._garbage_push_delay,
            search_depth=search_depth,
            beam_width=beam_width,
            max_len=max_len,
        )

        if action_idx < 0:
            flush(episode_buf, is_death=True)
            episode_buf = []
            deaths += 1
            time_step = env.reset()
            continue

        sequence = sequence.astype(np.int64)

        if not np.any(np.all(valid_sequences == sequence[None, :], axis=-1)):
            unmatched += 1
            time_step = env._step(sequence)
            if time_step.is_last():
                flush(episode_buf, is_death=True)
                episode_buf = []
                deaths += 1
                time_step = env.reset()
            continue

        mask = _build_mask(sequence, valid_sequences, max_len, key_dim)
        time_step = env._step(sequence)
        reward = float(time_step.reward["total_reward"])
        done = bool(time_step.is_last())

        episode_buf.append(
            (board, pieces, bcg, sequence, mask, np.float32(search_depth), reward, done)
        )
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if done:
            flush(episode_buf, is_death=True)
            episode_buf = []
            deaths += 1
            time_step = env.reset()

        if (step + 1) % log_every == 0:
            print(
                f"Step {step + 1}/{num_steps} | "
                f"transitions={len(transitions)} unmatched={unmatched} "
                f"deaths={deaths} max_b2b={max_b2b}",
                flush=True,
            )

    flush(episode_buf, is_death=False)
    return transitions, unmatched, deaths, max_b2b


class Pretrainer:
    def __init__(
        self,
        dataset_path="../tetris_expert_dataset_b2b",
        queue_size=5,
        max_len=15,
        key_dim=12,
        max_height=18,
        max_holes=50,
        max_steps_env=9999999,
        search_depth=7,
        beam_width=96,
        garbage_chance=0.15,
        garbage_min=1,
        garbage_max=4,
        garbage_push_delay=1,
        num_row_tiers=2,
        death_trim_count=20,
        gamma=0.99,
    ):
        self._dataset_path = dataset_path
        self._queue_size = queue_size
        self._max_len = max_len
        self._key_dim = key_dim
        self._max_height = max_height
        self._max_holes = max_holes
        self._max_steps_env = max_steps_env
        self._search_depth = search_depth
        self._beam_width = beam_width
        self._garbage_chance = garbage_chance
        self._garbage_min = garbage_min
        self._garbage_max = garbage_max
        self._garbage_push_delay = garbage_push_delay
        self._num_row_tiers = num_row_tiers
        self._death_trim_count = death_trim_count
        self._gamma = gamma
        self._scc = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def _generate_dataset(self, num_steps, seed):
        existing_count = 0
        existing = None
        if os.path.exists(self._dataset_path):
            try:
                existing_ds = tf.data.Dataset.load(self._dataset_path)
                existing = {
                    k: v.numpy()
                    for k, v in next(iter(existing_ds.batch(10_000_000))).items()
                }
                existing_count = len(existing["actions"])
                if "returns" not in existing:
                    print(
                        "Existing dataset has no `returns` field — starting fresh "
                        "(value pretraining requires returns).",
                        flush=True,
                    )
                    existing = None
                    existing_count = 0
                else:
                    print(
                        f"Found existing dataset with {existing_count} transitions",
                        flush=True,
                    )
            except Exception:
                print("Existing dataset load failed, starting fresh", flush=True)

        new_transitions, unmatched, deaths, max_b2b = _collect(
            seed=seed + existing_count,
            num_steps=num_steps,
            search_depth=self._search_depth,
            beam_width=self._beam_width,
            queue_size=self._queue_size,
            max_len=self._max_len,
            key_dim=self._key_dim,
            max_height=self._max_height,
            max_holes=self._max_holes,
            max_steps_env=self._max_steps_env,
            garbage_chance=self._garbage_chance,
            garbage_min=self._garbage_min,
            garbage_max=self._garbage_max,
            garbage_push_delay=self._garbage_push_delay,
            num_row_tiers=self._num_row_tiers,
            death_trim_count=self._death_trim_count,
            gamma=self._gamma,
        )

        print(
            f"Collected {len(new_transitions)} transitions | "
            f"unmatched: {unmatched} | deaths: {deaths} | max_b2b: {max_b2b}",
            flush=True,
        )

        boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
        pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
        bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
        actions = np.stack([t[3] for t in new_transitions]).astype(np.int64)
        masks = np.stack([t[4] for t in new_transitions]).astype(bool)
        sample_weights = np.stack([t[5] for t in new_transitions]).astype(np.float32)
        returns = np.stack([t[6] for t in new_transitions]).astype(np.float32)

        if existing is not None:
            boards = np.concatenate([existing["boards"], boards])
            pieces = np.concatenate([existing["pieces"], pieces])
            bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
            actions = np.concatenate([existing["actions"], actions])
            masks = np.concatenate([existing["masks"], masks])
            existing_weights = existing.get(
                "sample_weights",
                np.ones(existing_count, dtype=np.float32),
            )
            sample_weights = np.concatenate([existing_weights, sample_weights])
            returns = np.concatenate([existing["returns"], returns])
            print(
                f"Combined: {existing_count} existing + {len(new_transitions)} new = "
                f"{len(actions)} total",
                flush=True,
            )

        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "boards": boards,
                "pieces": pieces,
                "b2b_combo_garbage": bcg,
                "actions": actions,
                "masks": masks,
                "sample_weights": sample_weights,
                "returns": returns,
            }
        )
        dataset.save(self._dataset_path)
        return tf.data.Dataset.load(self._dataset_path)

    def _load_dataset(self, batch_size, num_steps, seed):
        dataset = self._generate_dataset(num_steps=num_steps, seed=seed)

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
        returns = batch["returns"]

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
        num_steps=200_000,
        seed=0,
        p_checkpoint_manager=None,
        v_checkpoint_manager=None,
    ):
        dataset = self._load_dataset(
            batch_size=batch_size,
            num_steps=num_steps,
            seed=seed,
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                policy_loss, accuracy, value_loss = self._train_step(p_model, v_model, batch)
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
    key_dim = 12
    depth = 64
    max_len = 15
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = 512
    num_row_tiers = 2

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
            keras.Input(shape=(max_len,), dtype=tf.int64),
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
        p_checkpoint, "./pretrained_checkpoints", max_to_keep=3
    )
    if p_checkpoint_manager.latest_checkpoint:
        p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained policy checkpoint.", flush=True)

    v_checkpoint = tf.train.Checkpoint(model=v_model, optimizer=v_optimizer)
    v_checkpoint_manager = tf.train.CheckpointManager(
        v_checkpoint, "./pretrained_value_checkpoints", max_to_keep=3
    )
    if v_checkpoint_manager.latest_checkpoint:
        v_checkpoint.restore(v_checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained value checkpoint.", flush=True)

    pretrainer = Pretrainer(
        queue_size=queue_size,
        max_len=max_len,
        key_dim=key_dim,
        num_row_tiers=num_row_tiers,
    )

    pretrainer.train(
        p_model,
        v_model,
        epochs=10,
        batch_size=batch_size,
        num_steps=200_000,
        p_checkpoint_manager=p_checkpoint_manager,
        v_checkpoint_manager=v_checkpoint_manager,
    )


if __name__ == "__main__":
    main()
