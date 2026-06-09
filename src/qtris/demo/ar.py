import tensorflow as tf
from qtris.models.ar.model import PolicyModel
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Moves import Keys
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame

from qtris.demo.constants import PIECE_COLORS, READABLE_KEYS
from qtris.demo.panels import (
    MaxStatTracker,
    draw_bcg_panel,
    draw_board_area,
    draw_info_panel,
    draw_step_counter,
    run_replay,
)
from qtris.demo.rendering import (
    colorize_attention_scores,
    colorize_piece_sidebar,
    compute_bcg_heatmaps,
    draw_garbage_bar,
)
from qtris.demo.utils import load_checkpoint, load_piece_display, save_frames_as_video
import time

# Model params
num_envs = 1
piece_dim = 8
key_dim = 12
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.1
max_len = 15

num_steps = 500
queue_size = 5
max_holes = 100
max_height = 18


def main(args):
    p_model = PolicyModel(
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

    load_checkpoint(p_model, args.checkpoint)

    p_model(
        (
            tf.keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            tf.keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            tf.keras.Input(shape=(3,), dtype=tf.float32),
            tf.keras.Input(shape=(max_len,), dtype=tf.int64),
        )
    )

    p_model.summary()

    py_env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=num_steps,
        max_len=max_len,
        pathfinding=True,
        garbage_chance=0.15,
        garbage_min=1,
        garbage_max=4,
        seed=0,
        idx=0,
        num_row_tiers=2,
    )
    env = TFPyEnvironment(py_env)

    screen_w = 870
    screen_h = 800
    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tetris")
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 22)

    time_step = env.reset()

    frames = []
    attacks = []
    apps = []
    clears = []
    actions = []
    attack_rewards = []
    total_rewards = []
    garbage_pusheds = []
    current_b2b = []
    current_combo = []
    current_garbage = []
    valid_seq_counts = []
    max_valid_seq_counts = []
    max_valid_seqs = 0
    max_stats = []

    death = 0
    running_attacks = 0
    running_clears = 0
    stat_tracker = MaxStatTracker()

    piece_display = load_piece_display()

    def draw_bottom_panel(surface, ind):
        draw_info_panel(
            surface,
            font,
            small_font,
            screen_w,
            [
                f"Attack Reward: {attack_rewards[ind]:0.2f}",
                f"Total Reward: {total_rewards[ind]:0.2f}",
                f"Garbage Pushed: {int(garbage_pusheds[ind])}",
                f"Valid Seqs: {valid_seq_counts[ind]}",
                f"Max Valid Seqs: {max_valid_seq_counts[ind]}",
            ],
            [
                f"Attack: {int(attacks[ind])}",
                f"APP: {apps[ind]:0.2f}",
                f"Clear: {int(clears[ind])}",
                f"Current B2B: {current_b2b[ind]}",
                f"Current Combo: {current_combo[ind]}",
                f"Garbage Queue: {current_garbage[ind]}",
            ],
            max_stats[ind],
            actions[ind],
        )

    start = time.time()
    for t in range(num_steps):
        board = time_step.observation["board"]
        vis_board = time_step.observation.get("vis_board", None)
        b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
        pieces = time_step.observation["pieces"]
        valid_sequences = time_step.observation["sequences"]
        attack = time_step.reward["attack"].numpy()[0]
        clear = time_step.reward["clear"].numpy()[0]
        attack_reward = time_step.reward["attack_reward"].numpy()[0]
        total_reward = time_step.reward["total_reward"].numpy()[0]
        garbage_pushed = time_step.reward["garbage_pushed"].numpy()[0]

        current_b2b_val = py_env._scorer._b2b
        current_combo_val = py_env._scorer._combo
        current_garbage_val = py_env._get_total_garbage()
        max_stats.append(
            stat_tracker.update(current_b2b_val, current_combo_val, attack)
        )

        if time_step.is_last():
            death = t
            running_attacks = 0
            running_clears = 0
            stat_tracker.reset_episode()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # A reachable placement is any sequence row that is not all-PAD.
        num_valid_seqs = int(
            tf.reduce_sum(
                tf.cast(tf.reduce_any(valid_sequences != Keys.PAD, axis=-1), tf.int32)
            )
        )
        max_valid_seqs = max(max_valid_seqs, num_valid_seqs)

        key_sequence, log_prob, action_index, scores = p_model.predict(
            (board, pieces, b2b_combo_garbage),
            greedy=True,
            valid_sequences=valid_sequences,
            temperature=1.0,
        )

        pieces_array = pieces.numpy()
        if pieces_array.ndim > 1:
            pieces_array = pieces_array[0]

        bcg_colored_heatmaps = compute_bcg_heatmaps(scores)
        colored_scores = colorize_attention_scores(scores, pieces_array, PIECE_COLORS)
        colored_sidebar = colorize_piece_sidebar(
            piece_display, pieces_array, PIECE_COLORS
        )
        garbage_surface = draw_garbage_bar(py_env, height=24, width=10)

        screen.fill((0, 0, 0))
        draw_board_area(
            screen,
            board,
            vis_board,
            colored_sidebar,
            colored_scores,
            garbage_surface,
            PIECE_COLORS,
        )
        draw_bcg_panel(
            screen,
            small_font,
            bcg_colored_heatmaps,
            [current_b2b_val, current_combo_val, current_garbage_val],
        )
        draw_step_counter(screen, font, t + 1, num_steps)

        readable_action = "".join(
            [READABLE_KEYS.get(k, "") for k in key_sequence.numpy()[0]]
        )

        actions.append(readable_action)

        running_attacks += attack
        running_clears += clear

        attacks.append(running_attacks)
        apps.append(running_attacks / (t - death + 1))
        clears.append(running_clears)
        attack_rewards.append(attack_reward)
        total_rewards.append(total_reward)
        garbage_pusheds.append(garbage_pushed)
        current_b2b.append(current_b2b_val)
        current_combo.append(current_combo_val)
        current_garbage.append(current_garbage_val)
        valid_seq_counts.append(num_valid_seqs)
        max_valid_seq_counts.append(max_valid_seqs)

        time_step = env.step(key_sequence)

        draw_bottom_panel(screen, -1)

        pygame.display.update()

        frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

    time_taken = time.time() - start

    print(f"Time taken: {time_taken:3.2f} seconds")
    print(f"Steps: {num_steps} | Time per step: {(time_taken / num_steps):1.3f}")
    save_frames_as_video(frames, "Demo.mp4")

    run_replay(screen, font, frames, num_steps, draw_bottom_panel)
