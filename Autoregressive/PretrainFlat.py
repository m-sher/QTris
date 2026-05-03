from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from TetrisModelFlat import FlatPolicyModel, ValueModel
import multiprocessing
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

HARD_DROP_ID = Keys.HARD_DROP


def _available_cpus():
    """CPUs the current process can actually use, respecting affinity and cgroup quota."""
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = os.cpu_count() or 1
    else:
        affinity_count = os.cpu_count() or 1

    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return min(affinity_count, max(1, int(quota) // int(period)))
    except (FileNotFoundError, ValueError, OSError):
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota_us = int(f.read().strip())
        if quota_us > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period_us = int(f.read().strip())
            return min(affinity_count, max(1, quota_us // period_us))
    except (FileNotFoundError, ValueError, OSError):
        pass

    return affinity_count


def _play_one_game(args):
    """Worker entry point. Runs num_steps in one env (with resets on death).

    Per-episode discounted returns are computed in flush so kept transitions
    carry the discounted impact of the upcoming death penalty even though the
    trimmed final tail isn't added to the dataset itself.
    """
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
        gamma,
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
            board, pieces, bcg, action_idx, valid_mask, sample_weight, _r, _d = buf[t]
            transitions.append(
                (board, pieces, bcg, action_idx, valid_mask, sample_weight, returns_arr[t])
            )

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

        time_step = env._step(sequence)
        reward = float(time_step.reward["total_reward"])
        done = bool(time_step.is_last())

        episode_buf.append(
            (
                board,
                pieces,
                bcg,
                flat_action_idx,
                valid_mask,
                np.float32(search_depth),
                reward,
                done,
            )
        )
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if done:
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
        gamma=0.99,
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
        self._gamma = gamma
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
                self._gamma,
            )
            for i in range(num_games)
        ]

        all_transitions = []
        total_unmatched = 0
        total_deaths = 0
        global_max_b2b = 0

        n_workers = max(1, min(num_games, _available_cpus()))
        print(
            f"Collecting {num_games} games of {num_steps_per_game} steps "
            f"with {n_workers} workers (available CPUs: {_available_cpus()}).",
            flush=True,
        )

        games_done = 0
        with multiprocessing.Pool(processes=n_workers) as pool:
            for transitions, unmatched, deaths, max_b2b in pool.imap_unordered(
                _play_one_game, args_list, chunksize=1
            ):
                all_transitions.extend(transitions)
                total_unmatched += unmatched
                total_deaths += deaths
                global_max_b2b = max(global_max_b2b, max_b2b)
                games_done += 1
                print(
                    f"Game {games_done}/{num_games} | "
                    f"this: steps={len(transitions)} unmatched={unmatched} "
                    f"deaths={deaths} max_b2b={max_b2b} | "
                    f"total: steps={len(all_transitions)} unmatched={total_unmatched} "
                    f"deaths={total_deaths} max_b2b={global_max_b2b}",
                    flush=True,
                )

        print(
            f"Collected {len(all_transitions)} transitions | "
            f"unmatched: {total_unmatched} | deaths: {total_deaths} | "
            f"max_b2b: {global_max_b2b}",
            flush=True,
        )

        boards = np.stack([t[0] for t in all_transitions]).astype(np.float32)
        pieces = np.stack([t[1] for t in all_transitions]).astype(np.int64)
        bcg = np.stack([t[2] for t in all_transitions]).astype(np.float32)
        action_indices = np.array([t[3] for t in all_transitions]).astype(np.int64)
        valid_masks = np.stack([t[4] for t in all_transitions]).astype(bool)
        sample_weights = np.array([t[5] for t in all_transitions]).astype(np.float32)
        returns = np.array([t[6] for t in all_transitions]).astype(np.float32)

        if existing is not None:
            boards = np.concatenate([existing["boards"], boards])
            pieces = np.concatenate([existing["pieces"], pieces])
            bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
            action_indices = np.concatenate(
                [existing["action_indices"], action_indices]
            )
            valid_masks = np.concatenate([existing["valid_masks"], valid_masks])
            existing_weights = existing.get(
                "sample_weights",
                np.ones(existing_count, dtype=np.float32),
            )
            sample_weights = np.concatenate([existing_weights, sample_weights])
            returns = np.concatenate([existing["returns"], returns])
            print(
                f"Combined: {existing_count} existing + {len(all_transitions)} new = "
                f"{len(action_indices)} total",
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
                "returns": returns,
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
        num_games=1000,
        num_steps_per_game=200,
        seed=0,
        p_checkpoint_manager=None,
        v_checkpoint_manager=None,
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
    depth = 64
    max_len = 15
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

    pretrainer = FlatPretrainer(
        queue_size=queue_size,
        max_len=max_len,
        num_row_tiers=num_row_tiers,
    )

    pretrainer.train(
        p_model,
        v_model,
        epochs=10,
        batch_size=batch_size,
        num_games=1000,
        num_steps_per_game=200,
        p_checkpoint_manager=p_checkpoint_manager,
        v_checkpoint_manager=v_checkpoint_manager,
    )


if __name__ == "__main__":
    main()
