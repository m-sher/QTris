import tensorflow as tf
from qtris.models.ar.model import PolicyModel
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Moves import Keys
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
import numpy as np

from qtris.demo.constants import PIECE_COLORS, READABLE_KEYS, BCG_LABELS
from qtris.demo.rendering import (
    compute_bcg_heatmaps,
    draw_garbage_bar,
    colorize_piece_sidebar,
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

    # Initialize pygame
    screen_w = 870
    screen_h = 800
    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tetris")
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 22)

    # BCG attention panel layout
    bcg_panel_x = 680
    bcg_label_y = 5
    bcg_heatmap_y = 30
    bcg_heatmap_w = 50
    bcg_heatmap_h = 120
    bcg_gap = 15

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

    death = 0
    running_attacks = 0
    running_clears = 0

    piece_display = load_piece_display()

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

        # Get current b2b, combo, garbage values from observation
        current_b2b_val = py_env._scorer._b2b
        current_combo_val = py_env._scorer._combo
        current_garbage_val = py_env._get_total_garbage()

        if time_step.is_last():
            death = t
            running_attacks = 0
            running_clears = 0

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

        # Handle pieces tensor shape (remove batch dimension if present)
        pieces_array = pieces.numpy()
        if pieces_array.ndim > 1:
            pieces_array = pieces_array[0]  # Remove batch dimension

        # Compute dominant piece for each board patch based on attention scores
        # scores is list of tensors (num_layers, batch_size, seq_len, num_pieces, num_patches)
        piece_attention = tf.reduce_sum(
            scores, axis=[0, 2]
        )  # Sum over layers and seq positions, keep pieces and patches
        # Slice out BCG tokens: keep only piece queries (:7) and patch keys (:60)
        piece_patch_attn = piece_attention[0, :7, :60]
        dominant_pieces = tf.argmax(
            piece_patch_attn, axis=0
        )  # Find dominant piece for each patch (shape: 60)
        dominant_grid = tf.reshape(dominant_pieces, (12, 5))  # Reshape to (12, 5) grid

        # Extract attention intensities for dominant pieces
        dominant_attention = tf.reduce_max(
            piece_patch_attn, axis=0
        )  # Get attention value for dominant piece (shape: 60)
        dominant_attention_grid = tf.reshape(
            dominant_attention, (12, 5)
        )  # Reshape to (12, 5) grid

        bcg_colored_heatmaps = compute_bcg_heatmaps(scores)

        # Normalize attention intensities for brightness modulation
        attention_min = tf.reduce_min(dominant_attention_grid)
        attention_max = tf.reduce_max(dominant_attention_grid)
        attention_normalized = (dominant_attention_grid - attention_min) / (
            attention_max - attention_min + 1e-8
        )

        # Colorize the scores display based on dominant pieces with intensity modulation
        all_PIECE_COLORS = PIECE_COLORS[pieces_array]  # Colors for all 7 pieces
        colored_scores = np.zeros((12, 5, 3), dtype=np.uint8)
        dominant_grid_np = dominant_grid.numpy()
        attention_np = attention_normalized.numpy()

        for r in range(12):  # 12 rows
            for c in range(5):  # 5 columns
                piece_idx = dominant_grid_np[r, c]
                intensity = attention_np[r, c]
                colored_scores[r, c] = (all_PIECE_COLORS[piece_idx] * intensity).astype(
                    np.uint8
                )

        colored_sidebar = colorize_piece_sidebar(
            piece_display, pieces_array, PIECE_COLORS
        )
        garbage_surface = draw_garbage_bar(py_env, height=24, width=10)

        screen.fill((0, 0, 0))

        board_surf = pygame.Surface((10, 24))
        piece_surf = pygame.Surface((5, 28))
        scores_surf = pygame.Surface((5, 12))
        garbage_surf = pygame.Surface(
            (garbage_surface.shape[1], garbage_surface.shape[0])
        )

        if vis_board is not None:
            colored_board = PIECE_COLORS[vis_board[0, ..., 0].numpy()]
            pygame.surfarray.blit_array(board_surf, colored_board.transpose(1, 0, 2))
        else:
            pygame.surfarray.blit_array(board_surf, board[0, ..., 0].numpy().T * 255)

        pygame.surfarray.blit_array(piece_surf, colored_sidebar.transpose(1, 0, 2))
        pygame.surfarray.blit_array(scores_surf, colored_scores.transpose(1, 0, 2))
        pygame.surfarray.blit_array(garbage_surf, garbage_surface.transpose(1, 0, 2))

        board_surf = pygame.transform.scale(board_surf, (250, 600))
        piece_surf = pygame.transform.scale(piece_surf, (125, 600))
        scores_surf = pygame.transform.scale(scores_surf, (250, 600))
        garbage_surf = pygame.transform.scale(
            garbage_surf, (25, 600)
        )  # Thinner garbage bar

        # Create board with border
        board_with_border = pygame.Surface((254, 604))  # 2 pixels border on each side
        board_with_border.fill((255, 255, 255))  # Black border
        board_with_border.blit(
            board_surf, (2, 2)
        )  # Blit board with 2px offset for border

        screen.blit(garbage_surf, (0, 0))  # Garbage bar on the left
        screen.blit(board_with_border, (25, 0))  # Board with border, shifted right less
        screen.blit(piece_surf, (285, 0))  # Piece sidebar adjusted position
        screen.blit(scores_surf, (415, 0))  # Scores adjusted position

        # BCG attention panel (b2b, combo, garbage attention over board patches)
        bcg_vals = [current_b2b_val, current_combo_val, current_garbage_val]
        for i in range(3):
            hx = bcg_panel_x + i * (bcg_heatmap_w + bcg_gap)
            label_text = small_font.render(
                f"{BCG_LABELS[i]}: {bcg_vals[i]}", True, (255, 255, 255)
            )
            screen.blit(label_text, (hx, bcg_label_y))
            bcg_surf = pygame.Surface((5, 12))
            pygame.surfarray.blit_array(
                bcg_surf, bcg_colored_heatmaps[i].transpose(1, 0, 2)
            )
            bcg_surf = pygame.transform.scale(bcg_surf, (bcg_heatmap_w, bcg_heatmap_h))
            screen.blit(bcg_surf, (hx, bcg_heatmap_y))

        # Add step counter in top left with black background (moved down to avoid slider collision)
        step_text = font.render(f"Step: {t + 1}/{num_steps}", True, (255, 255, 255))
        step_rect = step_text.get_rect()
        step_rect.topleft = (10, 25)
        # Create black background for step counter
        pygame.draw.rect(
            screen, (0, 0, 0), step_rect.inflate(10, 4)
        )  # Padding around text
        screen.blit(step_text, (10, 25))

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

        # Create black background for text area to prevent overlap
        text_bg_rect = pygame.Rect(
            0, 610, screen_w, 190
        )  # Black background for text area
        pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

        # Draw white dividing line in the middle
        pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

        # Render text below board with white color
        base_y = 615  # Start below the board area

        # LEFT HALF: Reward Information (single column)
        attack_rew_text = font.render(
            f"Attack Reward: {attack_reward:0.2f}", True, (255, 255, 255)
        )
        total_rew_text = font.render(
            f"Total Reward: {total_reward:0.2f}", True, (255, 255, 255)
        )
        garbage_push_text = font.render(
            f"Garbage Pushed: {int(garbage_pushed)}", True, (255, 255, 255)
        )
        valid_seq_text = font.render(
            f"Valid Seqs: {num_valid_seqs}", True, (255, 255, 255)
        )
        max_valid_seq_text = font.render(
            f"Max Valid Seqs: {max_valid_seqs}", True, (255, 255, 255)
        )

        # Position reward texts in left half (single column)
        screen.blit(attack_rew_text, (10, base_y))
        screen.blit(total_rew_text, (10, base_y + 20))
        screen.blit(garbage_push_text, (10, base_y + 40))
        screen.blit(valid_seq_text, (10, base_y + 60))
        screen.blit(max_valid_seq_text, (10, base_y + 80))

        # RIGHT HALF: Current State Information (single column)
        attack_text = font.render(f"Attack: {int(attacks[-1])}", True, (255, 255, 255))
        app_text = font.render(f"APP: {apps[-1]:0.2f}", True, (255, 255, 255))
        clear_text = font.render(f"Clear: {int(clears[-1])}", True, (255, 255, 255))
        current_b2b_text = font.render(
            f"Current B2B: {current_b2b_val}", True, (255, 255, 255)
        )
        current_combo_text = font.render(
            f"Current Combo: {current_combo_val}", True, (255, 255, 255)
        )
        current_garbage_text = font.render(
            f"Garbage Queue: {current_garbage_val}", True, (255, 255, 255)
        )
        action_text = font.render(f"Action: {actions[-1]}", True, (255, 255, 255))

        # Position state texts in right half (single column)
        screen.blit(attack_text, (345, base_y))
        screen.blit(app_text, (345, base_y + 20))
        screen.blit(clear_text, (345, base_y + 40))
        screen.blit(current_b2b_text, (345, base_y + 60))
        screen.blit(current_combo_text, (345, base_y + 80))
        screen.blit(current_garbage_text, (345, base_y + 100))
        screen.blit(action_text, (345, base_y + 120))

        pygame.display.update()

        frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

    time_taken = time.time() - start

    print(f"Time taken: {time_taken:3.2f} seconds")
    print(f"Steps: {num_steps} | Time per step: {(time_taken / num_steps):1.3f}")
    save_frames_as_video(frames, "Demo.mp4")

    slider = Slider(
        screen,
        x=10,
        y=5,
        width=585,
        height=10,
        min=0,
        max=num_steps - 1,
        step=1,
        colour=(125, 125, 125),
        handleColour=(50, 50, 50),
    )

    # Held in vars so pygame_widgets' WeakSet doesn't GC them (bare exprs vanish).
    _back_btn = Button(
        screen,
        605,
        0,
        28,
        20,
        text="<",
        fontSize=16,
        margin=0,
        onClick=lambda: slider.setValue(max(0, slider.getValue() - 1)),
    )

    _fwd_btn = Button(
        screen,
        637,
        0,
        28,
        20,
        text=">",
        fontSize=16,
        margin=0,
        onClick=lambda: slider.setValue(min(num_steps - 1, slider.getValue() + 1)),
    )

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((0, 0, 0))

        ind = slider.getValue()
        frame = frames[ind]

        pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))

        # Add step counter in top left for replay with black background (moved down to avoid slider collision)
        step_text = font.render(f"Step: {ind + 1}/{num_steps}", True, (255, 255, 255))
        step_rect = step_text.get_rect()
        step_rect.topleft = (10, 25)
        # Create black background for step counter
        pygame.draw.rect(
            screen, (0, 0, 0), step_rect.inflate(10, 4)
        )  # Padding around text
        screen.blit(step_text, (10, 25))

        pygame_widgets.update(events)

        # Create black background for text area in replay mode to prevent overlap
        text_bg_rect = pygame.Rect(
            0, 610, screen_w, 190
        )  # Black background for text area
        pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

        # Draw white dividing line in the middle
        pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

        # Render text below board for replay with white color
        base_y = 615  # Start below the board area

        # LEFT HALF: Reward Information (single column)
        attack_rew_text = font.render(
            f"Attack Reward: {attack_rewards[ind]:0.2f}", True, (255, 255, 255)
        )
        total_rew_text = font.render(
            f"Total Reward: {total_rewards[ind]:0.2f}", True, (255, 255, 255)
        )
        garbage_push_text = font.render(
            f"Garbage Pushed: {int(garbage_pusheds[ind])}", True, (255, 255, 255)
        )
        valid_seq_text = font.render(
            f"Valid Seqs: {valid_seq_counts[ind]}", True, (255, 255, 255)
        )
        max_valid_seq_text = font.render(
            f"Max Valid Seqs: {max_valid_seq_counts[ind]}", True, (255, 255, 255)
        )

        # Position reward texts in left half (single column)
        screen.blit(attack_rew_text, (10, base_y))
        screen.blit(total_rew_text, (10, base_y + 20))
        screen.blit(garbage_push_text, (10, base_y + 40))
        screen.blit(valid_seq_text, (10, base_y + 60))
        screen.blit(max_valid_seq_text, (10, base_y + 80))

        # RIGHT HALF: Current State Information (single column)
        attack_text = font.render(f"Attack: {int(attacks[ind])}", True, (255, 255, 255))
        app_text = font.render(f"APP: {apps[ind]:0.2f}", True, (255, 255, 255))
        clear_text = font.render(f"Clear: {int(clears[ind])}", True, (255, 255, 255))
        current_b2b_text = font.render(
            f"Current B2B: {current_b2b[ind]}", True, (255, 255, 255)
        )
        current_combo_text = font.render(
            f"Current Combo: {current_combo[ind]}", True, (255, 255, 255)
        )
        current_garbage_text = font.render(
            f"Garbage Queue: {current_garbage[ind]}", True, (255, 255, 255)
        )
        action_text = font.render(f"Action: {actions[ind]}", True, (255, 255, 255))

        # Position state texts in right half (single column)
        screen.blit(attack_text, (345, base_y))
        screen.blit(app_text, (345, base_y + 20))
        screen.blit(clear_text, (345, base_y + 40))
        screen.blit(current_b2b_text, (345, base_y + 60))
        screen.blit(current_combo_text, (345, base_y + 80))
        screen.blit(current_garbage_text, (345, base_y + 100))
        screen.blit(action_text, (345, base_y + 120))

        pygame.display.update()
