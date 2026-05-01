from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from TetrisModelFlat import FlatPolicyModel
import multiprocessing
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

HARD_DROP_ID = Keys.HARD_DROP


def _play_one_game(args):
    (
        seed,
        num_steps,
        search_depth,
        beam_width,
        queue_size,
        max_len,
        max_height,
        max_holes,
        max_steps_env,
        garbage_chance,
        garbage_min,
        garbage_max,
        garbage_push_delay,
        num_row_tiers,
        death_trim_count,
    ) = args

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
    )

    time_step = env.reset()
    searcher = CB2BSearch()

    transitions = []
    episode_buf = []
    unmatched = 0
    deaths = 0
    max_b2b = 0

    def flush(buf, is_death):
        if is_death:
            kept = buf[:-death_trim_count] if len(buf) > death_trim_count else []
        else:
            kept = buf
        transitions.extend(kept)

    for _ in range(num_steps):
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

        matches = np.all(valid_sequences == sequence[None, :], axis=-1)
        if not np.any(matches):
            unmatched += 1
            time_step = env._step(sequence)
            if time_step.is_last():
                flush(episode_buf, is_death=True)
                episode_buf = []
                deaths += 1
                time_step = env.reset()
            continue

        flat_action_idx = int(np.argmax(matches))
        valid_mask = np.any(valid_sequences == HARD_DROP_ID, axis=-1)

        episode_buf.append(
            (board, pieces, bcg, flat_action_idx, valid_mask, np.float32(search_depth))
        )

        time_step = env._step(sequence)
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if time_step.is_last():
            flush(episode_buf, is_death=True)
            episode_buf = []
            deaths += 1
            time_step = env.reset()

    flush(episode_buf, is_death=False)
    return transitions, unmatched, deaths, max_b2b


class FlatPretrainer:
    def __init__(
        self,
        dataset_path="../tetris_expert_dataset_flat",
        queue_size=5,
        max_len=15,
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
    ):
        self._dataset_path = dataset_path
        self._queue_size = queue_size
        self._max_len = max_len
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
        self._scc = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def _generate_dataset(self, num_games, num_steps_per_game, seed):
        existing_count = 0
        existing = None
        if os.path.exists(self._dataset_path):
            try:
                existing_ds = tf.data.Dataset.load(self._dataset_path)
                existing = {
                    k: v.numpy()
                    for k, v in next(iter(existing_ds.batch(10_000_000))).items()
                }
                existing_count = len(existing["action_indices"])
                print(
                    f"Found existing dataset with {existing_count} transitions",
                    flush=True,
                )
            except Exception:
                print("Existing dataset load failed, starting fresh", flush=True)

        args_list = [
            (
                seed + existing_count + i,
                num_steps_per_game,
                self._search_depth,
                self._beam_width,
                self._queue_size,
                self._max_len,
                self._max_height,
                self._max_holes,
                self._max_steps_env,
                self._garbage_chance,
                self._garbage_min,
                self._garbage_max,
                self._garbage_push_delay,
                self._num_row_tiers,
                self._death_trim_count,
            )
            for i in range(num_games)
        ]

        all_transitions = []
        total_unmatched = 0
        total_deaths = 0

        games_done = 0
        with multiprocessing.Pool(
            processes=min(16, max(1, multiprocessing.cpu_count() - 1))
        ) as pool:
            for transitions, unmatched, deaths, max_b2b in pool.imap_unordered(
                _play_one_game, args_list, chunksize=1
            ):
                all_transitions.extend(transitions)
                total_unmatched += unmatched
                total_deaths += deaths
                games_done += 1
                print(
                    f"Game {games_done}/{num_games} | "
                    f"this: steps={len(transitions)} unmatched={unmatched} deaths={deaths} max_b2b={max_b2b} | "
                    f"total: steps={len(all_transitions)} unmatched={total_unmatched} deaths={total_deaths}",
                    flush=True,
                )

        print(
            f"Collected {len(all_transitions)} transitions | "
            f"unmatched: {total_unmatched} | deaths: {total_deaths}",
            flush=True,
        )

        boards = np.stack([t[0] for t in all_transitions]).astype(np.float32)
        pieces = np.stack([t[1] for t in all_transitions]).astype(np.int64)
        bcg = np.stack([t[2] for t in all_transitions]).astype(np.float32)
        action_indices = np.array([t[3] for t in all_transitions]).astype(np.int64)
        valid_masks = np.stack([t[4] for t in all_transitions]).astype(bool)
        sample_weights = np.array([t[5] for t in all_transitions]).astype(np.float32)

        if existing is not None:
            boards = np.concatenate([existing["boards"], boards])
            pieces = np.concatenate([existing["pieces"], pieces])
            bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
            action_indices = np.concatenate([existing["action_indices"], action_indices])
            valid_masks = np.concatenate([existing["valid_masks"], valid_masks])
            existing_weights = existing.get(
                "sample_weights",
                np.ones(existing_count, dtype=np.float32),
            )
            sample_weights = np.concatenate([existing_weights, sample_weights])
            print(
                f"Combined: {existing_count} existing + {len(all_transitions)} new = {len(action_indices)} total",
                flush=True,
            )

        if os.path.exists(self._dataset_path):
            shutil.rmtree(self._dataset_path)

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "boards": boards,
                "pieces": pieces,
                "b2b_combo_garbage": bcg,
                "action_indices": action_indices,
                "valid_masks": valid_masks,
                "sample_weights": sample_weights,
            }
        )
        dataset.save(self._dataset_path)
        return tf.data.Dataset.load(self._dataset_path)

    def _load_dataset(self, batch_size, num_games, num_steps_per_game, seed):
        dataset = self._generate_dataset(
            num_games=num_games,
            num_steps_per_game=num_steps_per_game,
            seed=seed,
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
    def _train_step(self, model, batch):
        board = batch["boards"]
        pieces = batch["pieces"]
        bcg = batch["b2b_combo_garbage"]
        action_indices = batch["action_indices"]
        valid_masks = batch["valid_masks"]
        sample_weights = batch["sample_weights"]

        with tf.GradientTape() as tape:
            logits = model(
                (board, pieces, bcg), training=True
            )
            masked_logits = tf.where(
                valid_masks, logits, tf.constant(-1e9, dtype=tf.float32)
            )
            per_sample_loss = self._scc(action_indices, masked_logits)
            loss = tf.math.divide_no_nan(
                tf.reduce_sum(per_sample_loss * sample_weights),
                tf.reduce_sum(sample_weights),
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        predicted = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, action_indices), tf.float32))

        return loss, accuracy

    def train(
        self,
        model,
        epochs=10,
        batch_size=256,
        num_games=2000,
        num_steps_per_game=200,
        seed=0,
        checkpoint_manager=None,
    ):
        dataset = self._load_dataset(
            batch_size=batch_size,
            num_games=num_games,
            num_steps_per_game=num_steps_per_game,
            seed=seed,
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            for step, batch in enumerate(dataset):
                loss, accuracy = self._train_step(model, batch)
                if step % 100 == 0:
                    print(
                        f"Step {step + 1} | Loss: {float(loss):2.3f} | "
                        f"Accuracy: {float(accuracy):1.3f}",
                        flush=True,
                    )
            if checkpoint_manager is not None:
                checkpoint_manager.save()


def main():
    piece_dim = 8
    depth = 64
    max_len = 15
    queue_size = 5
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.0
    batch_size = 512
    num_row_tiers = 2
    num_sequences = 160 * num_row_tiers

    model = FlatPolicyModel(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        num_sequences=num_sequences,
    )

    optimizer = keras.optimizers.Adam(3e-4)
    model.compile(optimizer=optimizer, jit_compile=True)
    print("Initialized model and optimizer.", flush=True)

    model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "./pretrained_flat_checkpoints", max_to_keep=3
    )
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored pretrained checkpoint.", flush=True)

    pretrainer = FlatPretrainer(
        queue_size=queue_size,
        max_len=max_len,
        num_row_tiers=num_row_tiers,
    )

    pretrainer.train(
        model,
        epochs=10,
        batch_size=batch_size,
        num_games=1000,
        num_steps_per_game=200,
        checkpoint_manager=checkpoint_manager,
    )


if __name__ == "__main__":
    main()
