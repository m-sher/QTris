from TetrisEnv.PyTetrisRunner import PyTetrisRunner
from TetrisEnv.Moves import Keys
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import tf_agents
import wandb
import time
import json

HARD_DROP_ID = Keys.HARD_DROP

piece_dim = 8
key_dim = 12
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.0
max_len = 15
num_row_tiers = 2
num_sequences = 160 * num_row_tiers

num_envs = 64
num_collection_steps = 2048
queue_size = 5
max_holes = 50
max_height = 18
max_steps = 9999
garbage_chance_min = 0.15
garbage_chance_max = 0.15
garbage_rows_min = 1
garbage_rows_max = 4

mini_batch_size = 1024
num_epochs = 5
learning_rate = 1e-4
temperature = 1.0

total_samples = num_collection_steps * num_envs
batches_per_epoch = total_samples // mini_batch_size

save_freq = 1
DATA_DIR = "./pretrain_data"
METRICS_PATH = "./pretrain_metrics.json"

@tf.function
def train_step(flat_model, batch):
    boards = tf.ensure_shape(batch["boards"], (mini_batch_size, 24, 10, 1))
    pieces = tf.ensure_shape(batch["pieces"], (mini_batch_size, queue_size + 2))
    b2b_combo_garbage = tf.ensure_shape(batch["b2b_combo_garbage"], (mini_batch_size, 3))
    valid_sequences = tf.ensure_shape(
        batch["valid_sequences"], (mini_batch_size, num_sequences, max_len)
    )
    action_indices = tf.ensure_shape(batch["action_indices"], (mini_batch_size,))

    valid_mask = tf.reduce_any(
        tf.equal(valid_sequences, tf.constant(HARD_DROP_ID, dtype=tf.int64)),
        axis=-1,
    )

    with tf.GradientTape() as tape:
        logits = flat_model(
            (boards, pieces, b2b_combo_garbage),
            training=True,
        )

        masked_logits = tf.where(
            valid_mask, logits, tf.constant(-1e9, dtype=tf.float32)
        )

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=action_indices, logits=masked_logits
        )
        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, flat_model.trainable_variables)
    flat_model.optimizer.apply_gradients(
        zip(gradients, flat_model.trainable_variables)
    )

    predicted = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, action_indices), tf.float32))

    return {"loss": loss, "accuracy": accuracy}


def collect_data():
    from TetrisModel import PolicyModel, ValueModel

    old_model = PolicyModel(
        batch_size=num_envs,
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

    old_model.build(
        input_shape=[
            (None, 24, 10, 1),
            (None, queue_size + 2),
            (None, 3),
            (None, max_len),
        ]
    )
    v_model.build(input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 3)])

    old_ckpt = tf.train.Checkpoint(model=old_model)
    old_mgr = tf.train.CheckpointManager(
        old_ckpt, "./checkpoints/policy_checkpoints_445k", max_to_keep=1
    )
    old_ckpt.restore(old_mgr.latest_checkpoint).expect_partial()
    print("Restored old autoregressive policy", flush=True)

    v_ckpt = tf.train.Checkpoint(model=v_model)
    v_mgr = tf.train.CheckpointManager(v_ckpt, "./checkpoints/value_checkpoints_445k", max_to_keep=1)
    v_ckpt.restore(v_mgr.latest_checkpoint).expect_partial()
    print("Restored value model", flush=True)

    old_model.summary()

    runner = PyTetrisRunner(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=max_steps,
        pathfinding=True,
        max_len=max_len,
        key_dim=key_dim,
        num_steps=num_collection_steps,
        num_envs=num_envs,
        garbage_chance_min=garbage_chance_min,
        garbage_chance_max=garbage_chance_max,
        garbage_rows_min=garbage_rows_min,
        garbage_rows_max=garbage_rows_max,
        p_model=old_model,
        v_model=v_model,
        temperature=temperature,
        seed=None,
        num_sequences=num_sequences,
        num_row_tiers=num_row_tiers,
    )

    print(f"Collecting {num_collection_steps} steps x {num_envs} envs...", flush=True)
    t0 = time.time()

    (
        all_boards,
        all_pieces,
        all_b2b_combo_garbage,
        _all_actions,
        _all_log_probs,
        _all_masks,
        all_valid_sequences,
        all_action_indices,
        _all_values,
        _all_last_values,
        all_attacks,
        _all_clears,
        _all_attack_reward,
        all_total_reward,
        all_dones,
        _all_garbage_pushed,
    ) = runner.collect_trajectory(render=False, progress=True)

    collect_time = time.time() - t0
    print(f"Collection done in {collect_time:.1f}s", flush=True)

    avg_attacks = float(tf.reduce_mean(tf.reduce_sum(all_attacks, axis=0)))
    avg_reward = float(tf.reduce_mean(tf.reduce_sum(all_total_reward, axis=0)))
    avg_deaths = float(tf.reduce_mean(tf.reduce_sum(all_dones, axis=0)))
    avg_pieces = float(tf.reduce_mean(
        num_collection_steps / (tf.reduce_sum(all_dones, axis=0) + 1)
    ))

    boards_flat = tf.reshape(all_boards, (-1, 24, 10, 1))
    pieces_flat = tf.reshape(all_pieces, (-1, (queue_size + 2)))
    b2b_combo_garbage_flat = tf.reshape(all_b2b_combo_garbage, (-1, 3))
    valid_sequences_flat = tf.reshape(
        all_valid_sequences, (-1, num_sequences, max_len)
    )
    action_indices_flat = tf.reshape(all_action_indices, (-1,))

    dataset = tf.data.Dataset.from_tensor_slices({
        "boards": boards_flat,
        "pieces": pieces_flat,
        "b2b_combo_garbage": b2b_combo_garbage_flat,
        "valid_sequences": valid_sequences_flat,
        "action_indices": action_indices_flat,
    })
    tf.data.Dataset.save(dataset, DATA_DIR)

    env_metrics = {
        "avg_attacks": avg_attacks,
        "avg_reward": avg_reward,
        "avg_deaths": avg_deaths,
        "avg_pieces": avg_pieces,
    }

    runner.env.close()

    with open(METRICS_PATH, "w") as f:
        json.dump(env_metrics, f)


def load_dataset():
    element_spec = {
        "boards": tf.TensorSpec(shape=(24, 10, 1), dtype=tf.float32),
        "pieces": tf.TensorSpec(shape=(queue_size + 2,), dtype=tf.int64),
        "b2b_combo_garbage": tf.TensorSpec(shape=(3,), dtype=tf.float32),
        "valid_sequences": tf.TensorSpec(shape=(num_sequences, max_len), dtype=tf.int64),
        "action_indices": tf.TensorSpec(shape=(), dtype=tf.int64),
    }
    return (
        tf.data.Dataset.load(DATA_DIR, element_spec=element_spec)
        .cache()
        .shuffle(buffer_size=total_samples)
        .batch(
            mini_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
            drop_remainder=True,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def train():
    from TetrisModelFlat import FlatPolicyModel

    dataset = load_dataset()
    with open(METRICS_PATH) as f:
        env_metrics = json.load(f)

    flat_model = FlatPolicyModel(
        batch_size=num_envs,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        num_sequences=num_sequences,
    )

    flat_model.build(
        input_shape=[
            (None, 24, 10, 1),
            (None, queue_size + 2),
            (None, 3),
        ]
    )

    flat_ckpt_weights = tf.train.Checkpoint(model=flat_model)
    flat_ckpt_weights_mgr = tf.train.CheckpointManager(
        flat_ckpt_weights, "./flat_head_policy_checkpoints", max_to_keep=1
    )
    if flat_ckpt_weights_mgr.latest_checkpoint:
        flat_ckpt_weights.restore(
            flat_ckpt_weights_mgr.latest_checkpoint
        ).expect_partial()
        print("Restored flat model from existing checkpoint", flush=True)
    else:
        ar_ckpt = tf.train.Checkpoint(model=flat_model)
        ar_mgr = tf.train.CheckpointManager(
            ar_ckpt, "./checkpoints/policy_checkpoints_445k", max_to_keep=1
        )
        ar_ckpt.restore(ar_mgr.latest_checkpoint).expect_partial()
        print("Restored flat model encoder from autoregressive checkpoint", flush=True)

    optimizer = keras.optimizers.Adam(learning_rate, clipnorm=0.5)
    flat_model.compile(optimizer=optimizer, jit_compile=True)

    flat_ckpt = tf.train.Checkpoint(model=flat_model, optimizer=optimizer)
    flat_mgr = tf.train.CheckpointManager(
        flat_ckpt, "./flat_head_policy_checkpoints", max_to_keep=3
    )

    print(
        f"Flat model trainable variables: {len(flat_model.trainable_variables)}",
        flush=True,
    )
    flat_model.summary()

    wandb_run = wandb.init(project="Tetris-Pretrain", config={
        "learning_rate": learning_rate,
        "mini_batch_size": mini_batch_size,
        "num_epochs": num_epochs,
        "num_envs": num_envs,
        "num_collection_steps": num_collection_steps,
        "total_samples": total_samples,
        "batches_per_epoch": batches_per_epoch,
        **env_metrics,
    })

    print(f"Training {num_epochs} epochs x {batches_per_epoch} batches...", flush=True)
    train_t0 = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        step = 0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            out = train_step(flat_model, batch)
            loss_val = float(out["loss"])
            acc_val = float(out["accuracy"])
            epoch_loss += loss_val
            epoch_acc += acc_val
            step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.4f}")

        avg_epoch_loss = epoch_loss / max(step, 1)
        avg_epoch_acc = epoch_acc / max(step, 1)

        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch_accuracy": avg_epoch_acc,
            "epoch": epoch,
        })

        if (epoch + 1) % save_freq == 0:
            flat_mgr.save()

    flat_mgr.save()
    train_time = time.time() - train_t0

    print(
        f"Training done in {train_time:.1f}s | "
        f"Final loss: {avg_epoch_loss:.4f} | Final acc: {avg_epoch_acc:.4f}",
        flush=True,
    )

    wandb_run.finish()


def main(argv):
    collect_data()
    train()


if __name__ == "__main__":
    tf_agents.system.multiprocessing.handle_main(main)
