from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Moves import Moves, Convert
from TetrisEnv.Pieces import PieceType
from TetrisModel import PolicyModel
import multiprocessing
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
from tqdm import tqdm
import time
import os


class Pretrainer:
    def __init__(self):
        self._new_pieces = {
            0: PieceType.N.value,
            1: PieceType.I.value,
            2: PieceType.T.value,
            3: PieceType.L.value,
            4: PieceType.J.value,
            5: PieceType.Z.value,
            6: PieceType.S.value,
            7: PieceType.O.value,
        }
        self.scc = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def _load_raw_data(self):
        """
        Loads the raw game data from text files.
        """
        raw_data = [[], []]
        for file in glob.glob(
            "E:\\MisaMino-Tetrio\\MisaMino\\tetris_ai\\logs\\game*.txt"
        ):
            with open(file) as f:
                contents = f.readlines()
                for line in contents:
                    raw_data[int(line[0])].append(line[2:])
            print(f"Loaded {len(contents)} lines from {file}", flush=True)
        return raw_data

    def _process_into_transitions(
        self, raw_data: list[list[str]]
    ) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        """
        Process the raw game data into transitions of ((state), (next_state)).
        Raw game data has the following fields in order, separated by '#':
        1. Player ID (removed by `_load_raw_data`)
        2. Active piece
        3. Hold piece
        4. Queue of next pieces
        5. Board (represented by a list of integers where each integer is
                  equivalent to a binary representation of the corresponding row)
        """
        transitions = []
        transitions_checked = 0
        last_time = time.time()
        for player_data in raw_data:
            for i in range(len(player_data) - 1):
                # Parse state
                split_results = player_data[i].strip().split("#")
                active = int(split_results[0])
                hold = int(split_results[1])
                queue = [int(piece) for piece in split_results[2].split(",")[:5]]
                piece_seq = np.array(
                    [self._new_pieces[piece] for piece in [active] + [hold] + queue],
                    dtype=np.int32,
                )
                board = np.array(
                    [
                        [int(bit) for bit in "{:032b}".format(int(row))[-10:][::-1]]
                        for row in split_results[3].split(",")[5:-4]
                    ],
                    dtype=np.int32,
                )
                state = (board, piece_seq)

                # Parse next_state
                next_split_results = player_data[i + 1].strip().split("#")
                next_active = int(next_split_results[0])
                next_hold = int(next_split_results[1])
                next_queue = [
                    int(piece) for piece in next_split_results[2].split(",")[:5]
                ]
                next_piece_seq = np.array(
                    [
                        self._new_pieces[piece]
                        for piece in [next_active] + [next_hold] + next_queue
                    ],
                    dtype=np.int32,
                )
                next_board = np.array(
                    [
                        [int(bit) for bit in "{:032b}".format(int(row))[-10:][::-1]]
                        for row in next_split_results[3].split(",")[5:-4]
                    ],
                    dtype=np.int32,
                )
                next_state = (next_board, next_piece_seq)

                transitions_checked += 1

                if self._is_candidate(state, next_state):
                    transitions.append((state, next_state))

                if time.time() - last_time > 5:
                    print(
                        f"\rtransitions checked: {transitions_checked} | Valid transitions: {len(transitions)}",
                        end="",
                        flush=True,
                    )
                    last_time = time.time()

        print(
            f"\rTotal transitions checked: {transitions_checked} | Total valid transitions: {len(transitions)}",
            flush=True,
        )

        return transitions

    def _is_candidate(
        self,
        state: tuple[np.ndarray, np.ndarray],
        next_state: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        """
        Determine if the transition from board_t to board_t1 is a candidate for training.
        This removes transitions between two episodes and transitions that accept garbage.
        """
        board, piece_seq = state
        next_board, next_piece_seq = next_state

        # Check that the queue cycles as expected
        # Piece sequence is [active, hold, next1, next2, next3, next4, next5]
        if not (
            np.array_equal(piece_seq[3:], next_piece_seq[2:-1])  # Queue cycled once
            or (
                piece_seq[1] == 0  # Hold started empty
                and next_piece_seq[1] != 0  # Hold is now occupied
                and np.array_equal(piece_seq[4:], next_piece_seq[2:-2])
            )
        ):  # Queue cycled twice
            return False

        # Check that the board changes in an expected way
        cell_count_diff = np.sum(next_board) - np.sum(board)
        if not (
            cell_count_diff == 4  # Placed a piece without clearing lines
            or cell_count_diff == 4 - 10  # Placed a piece and cleared one line
            or cell_count_diff == 4 - 20  # Placed a piece and cleared two lines
            or cell_count_diff == 4 - 30  # Placed a piece and cleared three lines
            or cell_count_diff == 4 - 40
        ):  # Placed a piece and cleared four lines
            return False

        return True

    def _find_action(
        self,
        transition: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        test_env: PyTetrisEnv,
    ) -> tuple[int, int, int]:
        """
        Determine the action taken to transition from state_t to state_t1.
        """
        state, next_state = transition

        board, piece_seq = state
        next_board, next_piece_seq = next_state

        # Determine whether hold was used
        if piece_seq[1] != next_piece_seq[1]:
            hold = 1
        else:
            hold = 0

        # Determine the action(s) that cause the transition
        # Checking non-spins first to avoid unnecessary keypresses with spins
        for spin in range(len(Moves._spins)):
            for standard in range(len(Moves._standards)):
                action = {"hold": hold, "standard": standard, "spin": spin}
                active_piece = test_env._spawn_piece(PieceType(piece_seq[0]))
                hold_piece = PieceType(piece_seq[1])
                queue = [PieceType(piece) for piece in piece_seq[2:]]
                _, _, sim_board, _, _, _ = test_env._execute_action(
                    board, active_piece, hold_piece, queue, action
                )
                if np.array_equal(sim_board, next_board):
                    return board, piece_seq, (hold, standard, spin)

        # If no action is found, return action to filter
        return board, piece_seq, (-1, -1, -1)

    def process_single_transition(self, transition):
        return self._find_action(transition, self.test_env)

    def _generate_dataset(self) -> tf.data.Dataset:
        """
        Generate a dataset of transitions from the raw game data.
        """

        # Initialize test environment
        self.test_env = PyTetrisEnv(queue_size=5, max_holes=100, seed=123, idx=0)

        # Load transitions
        raw_data = self._load_raw_data()
        transition_df = self._process_into_transitions(raw_data)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        self.process_single_transition, transition_df, chunksize=512
                    ),
                    total=len(transition_df),
                    desc="Processing transitions",
                )
            )

        # Convert results from list of tuples to tuple of lists
        repacked = tuple(map(list, zip(*results)))

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(repacked).filter(
            lambda board, piece_seq, action: tf.reduce_all(action[1] != (-1, -1, -1))
        )

        # Save dataset and ensure it can be loaded
        dataset.save("../tetris_expert_dataset")
        dataset = tf.data.Dataset.load("../tetris_expert_dataset")

        return dataset

    @tf.function
    def _get_key_sequence(self, separate_action):
        action = tf.gather_nd(Convert.to_ind, separate_action)
        key_sequence = tf.gather(Convert.to_sequence, action)
        return key_sequence

    def _load_dataset(self, batch_size: int | None = 1024) -> tf.data.Dataset:
        """
        Load the dataset from disk if it exists, otherwise generate it.
        """

        # Check if dataset exists
        if not os.path.exists("../tetris_expert_dataset"):
            print("Dataset not found. Generating dataset...", flush=True)
            # Generate dataset
            dataset = self._generate_dataset()
        else:
            try:
                dataset = tf.data.Dataset.load("../tetris_expert_dataset")
                print("Dataset loaded successfully.", flush=True)
            except Exception:
                print("Dataset loading failed. Generating dataset...", flush=True)
                # Generate dataset
                dataset = self._generate_dataset()

        dataset = (
            dataset.map(
                lambda board, piece_seq, separate_action: {
                    "boards": tf.cast(board[..., None], tf.float32),
                    "pieces": tf.cast(piece_seq, tf.int64),
                    "actions": tf.cast(
                        self._get_key_sequence(separate_action), tf.int64
                    ),
                },
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            .cache()
            .shuffle(1000000)
        )

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
        key_sequence = batch["actions"]
        input_sequence = key_sequence[:, :-1]
        target_sequence = key_sequence[:, 1:]

        with tf.GradientTape() as tape:
            # Forward pass
            logits = model((board, piece_seq, input_sequence), training=True)
            # Compute loss
            loss = self.scc(target_sequence, logits)
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Compute accuracy
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.argmax(logits, axis=-1, output_type=tf.int64) == target_sequence,
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
    key_dim = 12
    depth = 64
    max_len = 9
    num_heads = 4
    num_layers = 4
    dropout_rate = 0.1
    batch_size = 1024

    # Initialize model and optimizer
    model = PolicyModel(
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

    optimizer = keras.optimizers.Adam(3e-5)
    model.compile(optimizer=optimizer, jit_compile=True)
    print("Initialized model and optimizer.", flush=True)

    dummy_board = tf.random.uniform((32, 24, 10, 1), dtype=tf.float32)
    dummy_pieces = tf.random.uniform((32, 7), dtype=tf.int32, minval=0, maxval=8)
    dummy_keys = tf.random.uniform((32, 9), dtype=tf.int32, minval=0, maxval=12)
    model((dummy_board, dummy_pieces, dummy_keys), training=False)
    model.summary()

    # Load checkpoint if it exists
    # Initialize checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # checkpoint_manager = tf.train.CheckpointManager(checkpoint, './policy_checkpoints', max_to_keep=3)
    # checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, "./pretrained_checkpoints", max_to_keep=3
    )
    # print("Restored checkpoint.", flush=True)

    pretrainer = Pretrainer()
    pretrainer.train(
        model, batch_size=batch_size, epochs=10, checkpoint_manager=checkpoint_manager
    )


if __name__ == "__main__":
    main()
