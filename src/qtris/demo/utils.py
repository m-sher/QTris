"""Shared utility functions for Tetris demos."""
import numpy as np
import tensorflow as tf


def load_checkpoint(model, checkpoint_path, max_to_keep=3):
    """Restore model weights from a tf.train.CheckpointManager directory.

    Returns True if a checkpoint was found and restored, False otherwise.
    """
    ckpt = tf.train.Checkpoint(model=model)
    mgr = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=max_to_keep)
    if mgr.latest_checkpoint is None:
        print(f"No checkpoint found in {checkpoint_path}, using random weights", flush=True)
        return False
    ckpt.restore(mgr.latest_checkpoint).expect_partial()
    print(f"Loaded checkpoint from {checkpoint_path}", flush=True)
    return True


def load_piece_display(path="PieceDisplay.npy"):
    """Load the (7, 4, 5) piece shape display array."""
    return np.load(path)


def save_frames_as_video(frames, output_path="Demo.mp4", fps=30, playback_fps=5, prompt=True):
    """Write recorded frames to an mp4 video via imageio.

    If prompt=True, asks the user before saving. Each frame is repeated
    (fps // playback_fps) times to achieve the target playback speed.
    """
    if prompt:
        if input("Save? ").lower() != "y":
            return False
    import imageio
    writer = imageio.get_writer(output_path, fps=fps)
    repeats = max(1, fps // playback_fps)
    for frame in frames:
        for _ in range(repeats):
            writer.append_data(frame)
    writer.close()
    print(f"Saved {len(frames)} frames to {output_path}", flush=True)
    return True
