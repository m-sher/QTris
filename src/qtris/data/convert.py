"""Convert pretraining datasets between flat and autoregressive formats.

Two directions are supported via ``--direction``:

  * ``flat_to_ar`` (default): scalar ``action_indices`` + 1D ``valid_masks``
    ``(160*num_row_tiers,)`` -> full key sequences ``(max_len,)`` + per-position
    masks ``(max_len, key_dim)``.
  * ``ar_to_flat``: full key sequences + per-position masks -> scalar
    ``action_indices`` + 1D ``valid_masks``.

Both directions rebuild the env from ``(board, pieces)`` to recover
``obs["sequences"]`` and use it as the bridge between the two representations.
All other fields (``boards``, ``pieces``, ``b2b_combo_garbage``,
``sample_weights``, ``returns``) are passed through unchanged.
"""

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Pieces import PieceType
from TetrisEnv.Moves import Keys
import numpy as np
import tensorflow as tf
import os
import shutil
import argparse


HARD_DROP_ID = Keys.HARD_DROP


def _build_ar_mask(sequence, valid_sequences, max_len, key_dim):
    """Per-position valid-next-token masks (mirrors DataGen.py)."""
    masks = np.zeros((max_len, key_dim), dtype=bool)
    for pos in range(1, max_len):
        prefix = sequence[:pos]
        match = np.all(valid_sequences[:, :pos] == prefix, axis=-1)
        if not match.any():
            continue
        next_tokens = valid_sequences[match, pos]
        masks[pos, np.unique(next_tokens)] = True
    return masks


def _reconstruct_valid_sequences(env, board, pieces, queue_size):
    """Reset env to (board, pieces) and return obs['sequences'].

    Only the fields that affect sequence enumeration (board, active piece,
    hold piece, queue) are restored — b2b/combo/garbage do not feed into the
    pathfinder.
    """
    env._board = board.copy()
    env._active_piece = env._spawn_piece(PieceType(int(pieces[0])))
    env._hold_piece = PieceType(int(pieces[1]))
    env._queue = [PieceType(int(pieces[j])) for j in range(2, 2 + queue_size)]
    return env._create_observation()["sequences"]


def flat_to_ar(flat_data, env, max_len, key_dim, queue_size):
    n = len(flat_data["action_indices"])
    actions_out = np.zeros((n, max_len), dtype=np.int64)
    masks_out = np.zeros((n, max_len, key_dim), dtype=bool)
    kept = np.zeros(n, dtype=bool)

    for i in range(n):
        board = flat_data["boards"][i, :, :, 0]
        pieces = flat_data["pieces"][i]
        action_idx = int(flat_data["action_indices"][i])

        valid_sequences = _reconstruct_valid_sequences(env, board, pieces, queue_size)
        chosen = valid_sequences[action_idx]
        if not np.any(chosen == HARD_DROP_ID):
            continue

        actions_out[i] = chosen
        masks_out[i] = _build_ar_mask(chosen, valid_sequences, max_len, key_dim)
        kept[i] = True

        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1}/{n} (skipped {i + 1 - int(kept[: i + 1].sum())})",
                flush=True,
            )

    skipped = int((~kept).sum())
    return {
        "boards": flat_data["boards"][kept],
        "pieces": flat_data["pieces"][kept],
        "b2b_combo_garbage": flat_data["b2b_combo_garbage"][kept],
        "actions": actions_out[kept],
        "masks": masks_out[kept],
        "sample_weights": flat_data["sample_weights"][kept],
        "returns": flat_data["returns"][kept],
    }, skipped


def ar_to_flat(ar_data, env, queue_size):
    n = len(ar_data["actions"])
    action_indices_out = np.zeros(n, dtype=np.int64)
    valid_masks_out = None
    kept = np.zeros(n, dtype=bool)

    for i in range(n):
        board = ar_data["boards"][i, :, :, 0]
        pieces = ar_data["pieces"][i]
        actions = ar_data["actions"][i]

        valid_sequences = _reconstruct_valid_sequences(env, board, pieces, queue_size)
        if valid_masks_out is None:
            valid_masks_out = np.zeros((n, valid_sequences.shape[0]), dtype=bool)

        matches = np.all(valid_sequences == actions[None, :], axis=-1)
        if not np.any(matches):
            continue

        action_indices_out[i] = int(np.argmax(matches))
        valid_masks_out[i] = np.any(valid_sequences == HARD_DROP_ID, axis=-1)
        kept[i] = True

        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1}/{n} (skipped {i + 1 - int(kept[: i + 1].sum())})",
                flush=True,
            )

    skipped = int((~kept).sum())
    return {
        "boards": ar_data["boards"][kept],
        "pieces": ar_data["pieces"][kept],
        "b2b_combo_garbage": ar_data["b2b_combo_garbage"][kept],
        "action_indices": action_indices_out[kept],
        "valid_masks": valid_masks_out[kept],
        "sample_weights": ar_data["sample_weights"][kept],
        "returns": ar_data["returns"][kept],
    }, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert pretraining dataset between flat and autoregressive formats"
    )
    parser.add_argument(
        "--direction",
        choices=["flat_to_ar", "ar_to_flat"],
        default="flat_to_ar",
        help="Conversion direction (default: flat_to_ar)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input dataset path (defaults inferred from --direction)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset path (defaults inferred from --direction)",
    )
    args = parser.parse_args()

    default_paths = {
        "flat_to_ar": (
            "datasets/tetris_expert_dataset_flat",
            "datasets/tetris_expert_dataset_b2b",
        ),
        "ar_to_flat": (
            "datasets/tetris_expert_dataset_b2b",
            "datasets/tetris_expert_dataset_flat",
        ),
    }
    in_default, out_default = default_paths[args.direction]
    in_path = args.input or in_default
    out_path = args.output or out_default

    max_len = 15
    key_dim = 12
    queue_size = 5
    num_row_tiers = 2

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"No dataset at {in_path}")

    print(f"Loading dataset from {in_path}...", flush=True)
    dataset = tf.data.Dataset.load(in_path)
    data = {
        k: v.numpy()
        for k, v in next(iter(dataset.batch(10_000_000))).items()
    }
    sample_key_in = "action_indices" if args.direction == "flat_to_ar" else "actions"
    if sample_key_in not in data:
        raise ValueError(
            f"Input at {in_path} is missing required field '{sample_key_in}' "
            f"for direction '{args.direction}'. Did you pick the wrong direction?"
        )
    print(f"Loaded {len(data[sample_key_in])} transitions", flush=True)

    env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=50,
        max_height=18,
        max_steps=9999999,
        max_len=max_len,
        pathfinding=True,
        seed=0,
        idx=0,
        num_row_tiers=num_row_tiers,
        gamma=0.99,
    )

    if args.direction == "flat_to_ar":
        out, skipped = flat_to_ar(data, env, max_len, key_dim, queue_size)
        sample_key_out = "actions"
    else:
        out, skipped = ar_to_flat(data, env, queue_size)
        sample_key_out = "action_indices"

    if skipped:
        print(
            f"Skipped {skipped} transitions with no representable label",
            flush=True,
        )

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    out_dataset = tf.data.Dataset.from_tensor_slices(out)
    out_dataset.save(out_path)
    print(
        f"Saved {len(out[sample_key_out])} transitions to {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
