import tensorflow as tf
from qtris.models.ar.model import PolicyModel
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Moves import Convert
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
import imageio
import numpy as np
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
max_height = 19

def main(args):
    left_path = str(args.left)
    left_pathfinding = True
    right_path = str(args.right)
    right_pathfinding = True


    p_model_left = PolicyModel(
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

    p_checkpoint_left = tf.train.Checkpoint(model=p_model_left)
    p_checkpoint_manager_left = tf.train.CheckpointManager(
        p_checkpoint_left, f"checkpoints/{left_path}", max_to_keep=3
    )
    p_checkpoint_left.restore(p_checkpoint_manager_left.latest_checkpoint).expect_partial()

    p_model_left.build(
        input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 3), (None, max_len)]
    )

    p_model_left.summary()

    py_env_left = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=num_steps,
        max_len=15,
        pathfinding=left_pathfinding,
        garbage_chance=0,
        garbage_min=0,
        garbage_max=0,
        seed=0,
        idx=0,
    )
    env_left = TFPyEnvironment(py_env_left)

    p_model_right = PolicyModel(
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

    p_checkpoint_right = tf.train.Checkpoint(model=p_model_right)
    p_checkpoint_manager_right = tf.train.CheckpointManager(
        p_checkpoint_right, f"checkpoints/{right_path}", max_to_keep=3
    )
    p_checkpoint_right.restore(
        p_checkpoint_manager_right.latest_checkpoint
    ).expect_partial()

    p_model_right.build(
        input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 3), (None, max_len)]
    )

    p_model_right.summary()

    py_env_right = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=num_steps,
        max_len=15,
        pathfinding=right_pathfinding,
        garbage_chance=0,
        garbage_min=0,
        garbage_max=0,
        seed=0,
        idx=0,
    )
    env_right = TFPyEnvironment(py_env_right)

    # Initialize pygame
    pygame.init()
    # Layout: left_scores(250) + left_garbage(25) + left_board(254) + left_pieces(125) + right_board(254) + right_pieces(125) + right_garbage(25) + right_scores(250) = 1308
    # Height: board(600) + text_area(200) = 800
    screen = pygame.display.set_mode((1308, 800))
    pygame.display.set_caption("Tetris VS")
    font = pygame.font.Font(None, 30)

    time_step_left = env_left.reset()
    time_step_right = env_right.reset()

    frames = []
    attacks_left = []
    apps_left = []
    clears_left = []
    actions_left = []
    b2b_rewards_left = []
    combo_rewards_left = []
    current_b2b_left = []
    current_combo_left = []
    height_penalties_left = []
    hole_penalties_left = []
    skyline_penalties_left = []
    bumpy_penalties_left = []
    death_penalties_left = []
    deaths_left = []

    attacks_right = []
    apps_right = []
    clears_right = []
    actions_right = []
    b2b_rewards_right = []
    combo_rewards_right = []
    current_b2b_right = []
    current_combo_right = []
    height_penalties_right = []
    hole_penalties_right = []
    skyline_penalties_right = []
    bumpy_penalties_right = []
    death_penalties_right = []
    deaths_right = []

    left_death = 0
    right_death = 0

    running_attacks_left = 0
    running_attacks_right = 0

    running_clears_left = 0
    running_clears_right = 0

    # Track garbage queue state before step for net attack calculation
    prev_garbage_total_left = 0
    prev_garbage_total_right = 0

    piece_display = np.load("PieceDisplay.npy")

    readable_keys = {
        1: "h",
        2: "l",
        3: "r",
        4: "L",
        5: "R",
        6: "c",
        7: "a",
        8: "1",
        9: "s",
        10: "H",
    }

    piece_colors = np.array(
        [
            [0, 0, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 127, 0],
            [255, 200, 0],
            [0, 255, 0],
            [255, 0, 255],
            [255, 0, 0],
            [127, 127, 127],
        ]
    )

    start = time.time()
    for t in range(num_steps):
        board_left = time_step_left.observation["board"]
        board_right = time_step_right.observation["board"]

        vis_board_left = time_step_left.observation.get("vis_board", None)
        vis_board_right = time_step_right.observation.get("vis_board", None)

        b2b_combo_left = time_step_left.observation["b2b_combo_garbage"]
        b2b_combo_right = time_step_right.observation["b2b_combo_garbage"]

        pieces_left = time_step_left.observation["pieces"]
        pieces_right = time_step_right.observation["pieces"]

        valid_sequences_left = time_step_left.observation["sequences"]
        valid_sequences_right = time_step_right.observation["sequences"]

        attack_left = time_step_left.reward["attack"].numpy()[0]
        attack_right = time_step_right.reward["attack"].numpy()[0]

        clear_left = time_step_left.reward["clear"].numpy()[0]
        clear_right = time_step_right.reward["clear"].numpy()[0]

        b2b_reward_left = time_step_left.reward["b2b_reward"].numpy()[0]
        b2b_reward_right = time_step_right.reward["b2b_reward"].numpy()[0]

        combo_reward_left = time_step_left.reward["combo_reward"].numpy()[0]
        combo_reward_right = time_step_right.reward["combo_reward"].numpy()[0]

        height_penalty_left = time_step_left.reward["height_penalty"].numpy()[0]
        height_penalty_right = time_step_right.reward["height_penalty"].numpy()[0]

        hole_penalty_left = time_step_left.reward["hole_penalty"].numpy()[0]
        hole_penalty_right = time_step_right.reward["hole_penalty"].numpy()[0]

        skyline_penalty_left = time_step_left.reward["skyline_penalty"].numpy()[0]
        skyline_penalty_right = time_step_right.reward["skyline_penalty"].numpy()[0]

        bumpy_penalty_left = time_step_left.reward["bumpy_penalty"].numpy()[0]
        bumpy_penalty_right = time_step_right.reward["bumpy_penalty"].numpy()[0]

        death_penalty_left = time_step_left.reward["death_penalty"].numpy()[0]
        death_penalty_right = time_step_right.reward["death_penalty"].numpy()[0]

        left_last = time_step_left.is_last()
        right_last = time_step_right.is_last()

        # Track if death occurred this step (for dense list)
        death_occurred_left = 0
        death_occurred_right = 0

        if left_last and t != 0:
            left_death = t
            running_attacks_left = 0
            running_clears_left = 0
            prev_garbage_total_left = 0
            death_occurred_left = 1
            # Reset right environment when left dies
            time_step_right = env_right.reset()
            # Mark right side "death boundary" for APP since its env was reset
            right_death = t
            running_attacks_right = 0
            running_clears_right = 0
            prev_garbage_total_right = 0

        if right_last and t != 0:
            right_death = t
            running_attacks_right = 0
            running_clears_right = 0
            prev_garbage_total_right = 0
            death_occurred_right = 1
            # Reset left environment when right dies
            time_step_left = env_left.reset()
            # Mark left side "death boundary" for APP since its env was reset
            left_death = t
            running_attacks_left = 0
            running_clears_left = 0
            prev_garbage_total_left = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if not left_pathfinding:
            valid_sequences_left = Convert.tf_to_sequence[None, ...]
        if not right_pathfinding:
            valid_sequences_right = Convert.tf_to_sequence[None, ...]

        key_sequence_left, log_probs_left, masks_left, scores_left = p_model_left.predict(
            (board_left, pieces_left, b2b_combo_left),
            greedy=True,
            valid_sequences=valid_sequences_left,
        )

        key_sequence_right, log_probs_right, masks_right, scores_right = (
            p_model_right.predict(
                (board_right, pieces_right, b2b_combo_right),
                greedy=True,
                valid_sequences=valid_sequences_right,
            )
        )

        # Get current b2b and combo values from environment
        current_b2b_val_left = py_env_left._scorer._b2b
        current_combo_val_left = py_env_left._scorer._combo
        current_b2b_val_right = py_env_right._scorer._b2b
        current_combo_val_right = py_env_right._scorer._combo

        # Cross-garbage: Calculate net attack after garbage cancellation
        # Use garbage queue state from BEFORE the previous step (when this attack was generated)
        # Left player's attack - calculate leftover after canceling their own garbage
        if attack_left > 0:
            # Net attack is what's left after canceling own garbage from before step
            net_attack_left = max(0, int(attack_left) - prev_garbage_total_left)
            if net_attack_left > 0:
                empty_column = py_env_right._random.randint(0, 9)
                py_env_right._garbage_queue.append((net_attack_left, empty_column))

        # Right player's attack - calculate leftover after canceling their own garbage
        if attack_right > 0:
            # Net attack is what's left after canceling own garbage from before step
            net_attack_right = max(0, int(attack_right) - prev_garbage_total_right)
            if net_attack_right > 0:
                empty_column = py_env_left._random.randint(0, 9)
                py_env_left._garbage_queue.append((net_attack_right, empty_column))

        # Handle pieces tensor shape - LEFT
        pieces_array_left = pieces_left.numpy()
        if pieces_array_left.ndim > 1:
            pieces_array_left = pieces_array_left[0]  # Remove batch dimension

        # Compute dominant piece for each board patch based on attention scores - LEFT
        # scores is list of tensors (num_layers, batch_size, seq_len, num_pieces, num_patches)
        piece_attention_left = tf.reduce_sum(
            scores_left, axis=[0, 2]
        )  # Sum over layers and seq positions, keep pieces and patches
        dominant_pieces_left = tf.argmax(
            piece_attention_left[0], axis=0
        )  # Find dominant piece for each patch (shape: 60)
        dominant_grid_left = tf.reshape(
            dominant_pieces_left, (12, 5)
        )  # Reshape to (12, 5) grid

        # Extract attention intensities for dominant pieces - LEFT
        dominant_attention_left = tf.reduce_max(
            piece_attention_left[0], axis=0
        )  # Get attention value for dominant piece (shape: 60)
        dominant_attention_grid_left = tf.reshape(
            dominant_attention_left, (12, 5)
        )  # Reshape to (12, 5) grid

        # Normalize attention intensities for brightness modulation - LEFT
        attention_min_left = tf.reduce_min(dominant_attention_grid_left)
        attention_max_left = tf.reduce_max(dominant_attention_grid_left)
        attention_normalized_left = (dominant_attention_grid_left - attention_min_left) / (
            attention_max_left - attention_min_left + 1e-8
        )

        # Colorize the scores display based on dominant pieces with intensity modulation - LEFT
        all_piece_colors_left = piece_colors[pieces_array_left]  # Colors for all 7 pieces
        colored_scores_left = np.zeros((12, 5, 3), dtype=np.uint8)
        dominant_grid_np_left = dominant_grid_left.numpy()
        attention_np_left = attention_normalized_left.numpy()

        for r in range(12):  # 12 rows
            for c in range(5):  # 5 columns
                piece_idx = dominant_grid_np_left[r, c]
                intensity = attention_np_left[r, c]
                colored_scores_left[r, c] = (
                    all_piece_colors_left[piece_idx] * intensity
                ).astype(np.uint8)

        # Get piece display for all 7 pieces - LEFT
        piece_sidebar_left = piece_display[pieces_array_left].reshape((28, 5))

        # Colorize the piece display sidebar - LEFT
        piece_type_colors_left = piece_colors[pieces_array_left]
        colored_sidebar_left = np.zeros((28, 5, 3), dtype=np.uint8)
        for i in range(7):  # 7 pieces: active, hold, 5 queue
            for r in range(4 * i, 4 * i + 4):  # 4 rows per piece
                for c in range(5):  # 5 columns
                    colored_sidebar_left[r, c] = (
                        piece_type_colors_left[i] * piece_sidebar_left[r, c]
                    ).astype(np.uint8)

        # Handle pieces tensor shape - RIGHT
        pieces_array_right = pieces_right.numpy()
        if pieces_array_right.ndim > 1:
            pieces_array_right = pieces_array_right[0]  # Remove batch dimension

        # Compute dominant piece for each board patch based on attention scores - RIGHT
        piece_attention_right = tf.reduce_sum(
            scores_right, axis=[0, 2]
        )  # Sum over layers and seq positions, keep pieces and patches
        dominant_pieces_right = tf.argmax(
            piece_attention_right[0], axis=0
        )  # Find dominant piece for each patch (shape: 60)
        dominant_grid_right = tf.reshape(
            dominant_pieces_right, (12, 5)
        )  # Reshape to (12, 5) grid

        # Extract attention intensities for dominant pieces - RIGHT
        dominant_attention_right = tf.reduce_max(
            piece_attention_right[0], axis=0
        )  # Get attention value for dominant piece (shape: 60)
        dominant_attention_grid_right = tf.reshape(
            dominant_attention_right, (12, 5)
        )  # Reshape to (12, 5) grid

        # Normalize attention intensities for brightness modulation - RIGHT
        attention_min_right = tf.reduce_min(dominant_attention_grid_right)
        attention_max_right = tf.reduce_max(dominant_attention_grid_right)
        attention_normalized_right = (
            dominant_attention_grid_right - attention_min_right
        ) / (attention_max_right - attention_min_right + 1e-8)

        # Colorize the scores display based on dominant pieces with intensity modulation - RIGHT
        all_piece_colors_right = piece_colors[pieces_array_right]  # Colors for all 7 pieces
        colored_scores_right = np.zeros((12, 5, 3), dtype=np.uint8)
        dominant_grid_np_right = dominant_grid_right.numpy()
        attention_np_right = attention_normalized_right.numpy()

        for r in range(12):  # 12 rows
            for c in range(5):  # 5 columns
                piece_idx = dominant_grid_np_right[r, c]
                intensity = attention_np_right[r, c]
                colored_scores_right[r, c] = (
                    all_piece_colors_right[piece_idx] * intensity
                ).astype(np.uint8)

        # Get piece display for all 7 pieces - RIGHT
        piece_sidebar_right = piece_display[pieces_array_right].reshape((28, 5))

        # Colorize the piece display sidebar - RIGHT
        piece_type_colors_right = piece_colors[pieces_array_right]
        colored_sidebar_right = np.zeros((28, 5, 3), dtype=np.uint8)
        for i in range(7):  # 7 pieces: active, hold, 5 queue
            for r in range(4 * i, 4 * i + 4):  # 4 rows per piece
                for c in range(5):  # 5 columns
                    colored_sidebar_right[r, c] = (
                        piece_type_colors_right[i] * piece_sidebar_right[r, c]
                    ).astype(np.uint8)

        # Create garbage queue visualization - LEFT
        garbage_queue_left = py_env_left._garbage_queue
        garbage_bar_width = 10  # Thinner width in pixels for the garbage bar
        garbage_bar_height = 24  # Height matches board height
        garbage_surface_left = np.zeros(
            (garbage_bar_height, garbage_bar_width, 3), dtype=np.uint8
        )

        # Draw garbage sections (bottom to top, with bottom being next to push) - LEFT
        current_row = garbage_bar_height - 1  # Start from bottom
        for i, (num_rows, empty_column) in enumerate(garbage_queue_left):
            # Draw red section for this garbage instance
            start_row = max(0, current_row - num_rows + 1)
            for row in range(start_row, current_row + 1):
                if row >= 0 and row < garbage_bar_height:
                    garbage_surface_left[row, :] = [255, 0, 0]  # Red color

            # Add separator line (1 pixel gap) between sections
            if (
                i < len(garbage_queue_left) - 1 and start_row > 0
            ):  # Not the last section and not at top
                current_row = start_row - 2  # Leave 1 pixel gap (black)
            else:
                current_row = start_row - 1
            if current_row < 0:
                break

        # Create garbage queue visualization - RIGHT
        garbage_queue_right = py_env_right._garbage_queue
        garbage_surface_right = np.zeros(
            (garbage_bar_height, garbage_bar_width, 3), dtype=np.uint8
        )

        # Draw garbage sections (bottom to top, with bottom being next to push) - RIGHT
        current_row = garbage_bar_height - 1  # Start from bottom
        for i, (num_rows, empty_column) in enumerate(garbage_queue_right):
            # Draw red section for this garbage instance
            start_row = max(0, current_row - num_rows + 1)
            for row in range(start_row, current_row + 1):
                if row >= 0 and row < garbage_bar_height:
                    garbage_surface_right[row, :] = [255, 0, 0]  # Red color

            # Add separator line (1 pixel gap) between sections
            if (
                i < len(garbage_queue_right) - 1 and start_row > 0
            ):  # Not the last section and not at top
                current_row = start_row - 2  # Leave 1 pixel gap (black)
            else:
                current_row = start_row - 1
            if current_row < 0:
                break

        screen.fill((0, 0, 0))

        # Create surfaces for LEFT player
        board_surf_left = pygame.Surface((10, 24))
        piece_surf_left = pygame.Surface((5, 28))
        scores_surf_left = pygame.Surface((5, 12))
        garbage_surf_left = pygame.Surface((garbage_bar_width, garbage_bar_height))

        if vis_board_left is not None:
            colored_board_left = piece_colors[vis_board_left[0, ..., 0].numpy()]
            pygame.surfarray.blit_array(
                board_surf_left, colored_board_left.transpose(1, 0, 2)
            )
        else:
            pygame.surfarray.blit_array(
                board_surf_left, board_left[0, ..., 0].numpy().T * 255
            )

        pygame.surfarray.blit_array(
            piece_surf_left, colored_sidebar_left.transpose(1, 0, 2)
        )
        pygame.surfarray.blit_array(
            scores_surf_left, colored_scores_left.transpose(1, 0, 2)
        )
        pygame.surfarray.blit_array(
            garbage_surf_left, garbage_surface_left.transpose(1, 0, 2)
        )

        board_surf_left = pygame.transform.scale(board_surf_left, (250, 600))
        piece_surf_left = pygame.transform.scale(piece_surf_left, (125, 600))
        scores_surf_left = pygame.transform.scale(scores_surf_left, (250, 600))
        garbage_surf_left = pygame.transform.scale(garbage_surf_left, (25, 600))

        # Create surfaces for RIGHT player
        board_surf_right = pygame.Surface((10, 24))
        piece_surf_right = pygame.Surface((5, 28))
        scores_surf_right = pygame.Surface((5, 12))
        garbage_surf_right = pygame.Surface((garbage_bar_width, garbage_bar_height))

        if vis_board_right is not None:
            colored_board_right = piece_colors[vis_board_right[0, ..., 0].numpy()]
            pygame.surfarray.blit_array(
                board_surf_right, colored_board_right.transpose(1, 0, 2)
            )
        else:
            pygame.surfarray.blit_array(
                board_surf_right, board_right[0, ..., 0].numpy().T * 255
            )

        pygame.surfarray.blit_array(
            piece_surf_right, colored_sidebar_right.transpose(1, 0, 2)
        )
        pygame.surfarray.blit_array(
            scores_surf_right, colored_scores_right.transpose(1, 0, 2)
        )
        pygame.surfarray.blit_array(
            garbage_surf_right, garbage_surface_right.transpose(1, 0, 2)
        )

        board_surf_right = pygame.transform.scale(board_surf_right, (250, 600))
        piece_surf_right = pygame.transform.scale(piece_surf_right, (125, 600))
        scores_surf_right = pygame.transform.scale(scores_surf_right, (250, 600))
        garbage_surf_right = pygame.transform.scale(garbage_surf_right, (25, 600))

        # Create boards with borders
        board_with_border_left = pygame.Surface((254, 604))  # 2 pixels border on each side
        board_with_border_left.fill((255, 255, 255))  # White border
        board_with_border_left.blit(board_surf_left, (2, 2))

        board_with_border_right = pygame.Surface((254, 604))
        board_with_border_right.fill((255, 255, 255))  # White border
        board_with_border_right.blit(board_surf_right, (2, 2))

        # Layout: left_scores(250) + left_garbage(25) + left_board(254) + left_pieces(125) + right_board(254) + right_pieces(125) + right_garbage(25) + right_scores(250)
        x_pos = 0
        screen.blit(scores_surf_left, (x_pos, 0))
        x_pos += 250
        screen.blit(garbage_surf_left, (x_pos, 0))
        x_pos += 25
        screen.blit(board_with_border_left, (x_pos, 0))
        x_pos += 254
        screen.blit(piece_surf_left, (x_pos, 0))
        x_pos += 125
        screen.blit(board_with_border_right, (x_pos, 0))
        x_pos += 254
        screen.blit(piece_surf_right, (x_pos, 0))
        x_pos += 125
        screen.blit(garbage_surf_right, (x_pos, 0))
        x_pos += 25
        screen.blit(scores_surf_right, (x_pos, 0))

        # Add step counter in top left with black background
        step_text = font.render(f"Step: {t + 1}/{num_steps}", True, (255, 255, 255))
        step_rect = step_text.get_rect()
        step_rect.topleft = (10, 25)
        pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))
        screen.blit(step_text, (10, 25))

        readable_action_left = "".join(
            [readable_keys.get(k, "") for k in key_sequence_left.numpy()[0]]
        )
        readable_action_right = "".join(
            [readable_keys.get(k, "") for k in key_sequence_right.numpy()[0]]
        )

        actions_left.append(readable_action_left)
        actions_right.append(readable_action_right)

        running_attacks_left += attack_left
        running_clears_left += clear_left
        running_attacks_right += attack_right
        running_clears_right += clear_right

        attacks_left.append(running_attacks_left)
        apps_left.append(running_attacks_left / (t - left_death + 1))
        clears_left.append(running_clears_left)
        b2b_rewards_left.append(b2b_reward_left)
        combo_rewards_left.append(combo_reward_left)
        current_b2b_left.append(current_b2b_val_left)
        current_combo_left.append(current_combo_val_left)
        height_penalties_left.append(height_penalty_left)
        hole_penalties_left.append(hole_penalty_left)
        skyline_penalties_left.append(skyline_penalty_left)
        bumpy_penalties_left.append(bumpy_penalty_left)
        death_penalties_left.append(death_penalty_left)
        deaths_left.append(death_occurred_left)

        attacks_right.append(running_attacks_right)
        apps_right.append(running_attacks_right / (t - right_death + 1))
        clears_right.append(running_clears_right)
        b2b_rewards_right.append(b2b_reward_right)
        combo_rewards_right.append(combo_reward_right)
        current_b2b_right.append(current_b2b_val_right)
        current_combo_right.append(current_combo_val_right)
        height_penalties_right.append(height_penalty_right)
        hole_penalties_right.append(hole_penalty_right)
        skyline_penalties_right.append(skyline_penalty_right)
        bumpy_penalties_right.append(bumpy_penalty_right)
        death_penalties_right.append(death_penalty_right)
        deaths_right.append(death_occurred_right)

        # Save garbage queue state BEFORE stepping for next iteration's net attack calculation
        prev_garbage_total_left = sum(
            num_rows for num_rows, _ in py_env_left._garbage_queue
        )
        prev_garbage_total_right = sum(
            num_rows for num_rows, _ in py_env_right._garbage_queue
        )

        # Step both environments (cross-garbage was already added before visualization)
        time_step_left = env_left.step(key_sequence_left)
        time_step_right = env_right.step(key_sequence_right)

        # Create black background for text area to prevent overlap
        text_bg_rect = pygame.Rect(0, 610, 1308, 190)  # Black background for text area
        pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

        # Draw white dividing line in the middle
        pygame.draw.line(screen, (255, 255, 255), (654, 610), (654, 800), 2)

        # Render text below board with white color
        base_y = 615  # Start below the board area

        # LEFT PLAYER Stats
        # Left side: Reward Information
        b2b_reward_text_left = font.render(
            f"B2B Reward: {b2b_reward_left:0.2f}", True, (255, 255, 255)
        )
        combo_reward_text_left = font.render(
            f"Combo Reward: {combo_reward_left:0.2f}", True, (255, 255, 255)
        )
        height_pen_text_left = font.render(
            f"Height Penalty: {height_penalty_left:0.2f}", True, (255, 255, 255)
        )
        hole_pen_text_left = font.render(
            f"Hole Penalty: {hole_penalty_left:0.2f}", True, (255, 255, 255)
        )
        skyline_pen_text_left = font.render(
            f"Skyline Penalty: {skyline_penalty_left:0.2f}", True, (255, 255, 255)
        )
        bumpy_pen_text_left = font.render(
            f"Bumpy Penalty: {bumpy_penalty_left:0.2f}", True, (255, 255, 255)
        )
        death_pen_text_left = font.render(
            f"Death Penalty: {death_penalty_left:0.2f}", True, (255, 255, 255)
        )

        # Position left player reward texts
        screen.blit(b2b_reward_text_left, (10, base_y))
        screen.blit(combo_reward_text_left, (10, base_y + 20))
        screen.blit(height_pen_text_left, (10, base_y + 40))
        screen.blit(hole_pen_text_left, (10, base_y + 60))
        screen.blit(skyline_pen_text_left, (10, base_y + 80))
        screen.blit(bumpy_pen_text_left, (10, base_y + 100))
        screen.blit(death_pen_text_left, (10, base_y + 120))

        # Right side of left half: Current State Information
        attack_text_left = font.render(
            f"Attack: {int(attacks_left[-1])}", True, (255, 255, 255)
        )
        app_text_left = font.render(f"APP: {apps_left[-1]:0.2f}", True, (255, 255, 255))
        clear_text_left = font.render(
            f"Clear: {int(clears_left[-1])}", True, (255, 255, 255)
        )
        current_b2b_text_left = font.render(
            f"Current B2B: {current_b2b_val_left}", True, (255, 255, 255)
        )
        current_combo_text_left = font.render(
            f"Current Combo: {current_combo_val_left}", True, (255, 255, 255)
        )
        action_text_left = font.render(f"Action: {actions_left[-1]}", True, (255, 255, 255))
        death_text_left = font.render(f"Deaths: {sum(deaths_left)}", True, (255, 255, 255))

        # Position left player state texts
        screen.blit(attack_text_left, (335, base_y))
        screen.blit(app_text_left, (335, base_y + 20))
        screen.blit(clear_text_left, (335, base_y + 40))
        screen.blit(current_b2b_text_left, (335, base_y + 60))
        screen.blit(current_combo_text_left, (335, base_y + 80))
        screen.blit(action_text_left, (335, base_y + 100))
        screen.blit(death_text_left, (335, base_y + 120))

        # RIGHT PLAYER Stats
        # Left side of right half: Reward Information
        b2b_reward_text_right = font.render(
            f"B2B Reward: {b2b_reward_right:0.2f}", True, (255, 255, 255)
        )
        combo_reward_text_right = font.render(
            f"Combo Reward: {combo_reward_right:0.2f}", True, (255, 255, 255)
        )
        height_pen_text_right = font.render(
            f"Height Penalty: {height_penalty_right:0.2f}", True, (255, 255, 255)
        )
        hole_pen_text_right = font.render(
            f"Hole Penalty: {hole_penalty_right:0.2f}", True, (255, 255, 255)
        )
        skyline_pen_text_right = font.render(
            f"Skyline Penalty: {skyline_penalty_right:0.2f}", True, (255, 255, 255)
        )
        bumpy_pen_text_right = font.render(
            f"Bumpy Penalty: {bumpy_penalty_right:0.2f}", True, (255, 255, 255)
        )
        death_pen_text_right = font.render(
            f"Death Penalty: {death_penalty_right:0.2f}", True, (255, 255, 255)
        )

        # Position right player reward texts
        screen.blit(b2b_reward_text_right, (670, base_y))
        screen.blit(combo_reward_text_right, (670, base_y + 20))
        screen.blit(height_pen_text_right, (670, base_y + 40))
        screen.blit(hole_pen_text_right, (670, base_y + 60))
        screen.blit(skyline_pen_text_right, (670, base_y + 80))
        screen.blit(bumpy_pen_text_right, (670, base_y + 100))
        screen.blit(death_pen_text_right, (670, base_y + 120))

        # Right side of right half: Current State Information
        attack_text_right = font.render(
            f"Attack: {int(attacks_right[-1])}", True, (255, 255, 255)
        )
        app_text_right = font.render(f"APP: {apps_right[-1]:0.2f}", True, (255, 255, 255))
        clear_text_right = font.render(
            f"Clear: {int(clears_right[-1])}", True, (255, 255, 255)
        )
        current_b2b_text_right = font.render(
            f"Current B2B: {current_b2b_val_right}", True, (255, 255, 255)
        )
        current_combo_text_right = font.render(
            f"Current Combo: {current_combo_val_right}", True, (255, 255, 255)
        )
        action_text_right = font.render(
            f"Action: {actions_right[-1]}", True, (255, 255, 255)
        )
        death_text_right = font.render(
            f"Deaths: {sum(deaths_right)}", True, (255, 255, 255)
        )

        # Position right player state texts
        screen.blit(attack_text_right, (995, base_y))
        screen.blit(app_text_right, (995, base_y + 20))
        screen.blit(clear_text_right, (995, base_y + 40))
        screen.blit(current_b2b_text_right, (995, base_y + 60))
        screen.blit(current_combo_text_right, (995, base_y + 80))
        screen.blit(action_text_right, (995, base_y + 100))
        screen.blit(death_text_right, (995, base_y + 120))

        pygame.display.update()

        frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

    time_taken = time.time() - start

    print(f"Time taken: {time_taken:3.2f} seconds")
    print(f"Steps: {num_steps} | Time per step: {(time_taken / num_steps):1.3f}")
    if input("Save? ").lower() == "y":
        actual_fps = 5
        writer = imageio.get_writer("Demo.mp4", fps=30)
        for frame in frames:
            for _ in range(30 // actual_fps):
                writer.append_data(frame)
        writer.close()

    slider = Slider(
        screen,
        x=10,
        y=5,
        width=1223,  # Adjusted width to match new screen size (1308 - 10 - 10 - 28 - 5 - 28 - 4)
        height=10,
        min=0,
        max=num_steps - 1,
        step=1,
        colour=(125, 125, 125),
        handleColour=(50, 50, 50),
    )

    back_btn = Button(
        screen,
        1243,  # Adjusted position for new screen width
        0,
        28,
        20,
        text="<",
        fontSize=16,
        margin=0,
        onClick=lambda: slider.setValue(max(0, slider.getValue() - 1)),
    )

    fwd_btn = Button(
        screen,
        1275,  # Adjusted position for new screen width
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
        pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))  # Padding around text
        screen.blit(step_text, (10, 25))

        pygame_widgets.update(events)

        # Create black background for text area in replay mode to prevent overlap
        text_bg_rect = pygame.Rect(0, 610, 1308, 190)  # Black background for text area
        pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

        # Draw white dividing line in the middle
        pygame.draw.line(screen, (255, 255, 255), (654, 610), (654, 800), 2)

        # Render text below board for replay with white color
        base_y = 615  # Start below the board area

        # LEFT PLAYER Stats - Replay
        # Left side: Reward Information
        b2b_reward_text_left = font.render(
            f"B2B Reward: {b2b_rewards_left[ind]:0.2f}", True, (255, 255, 255)
        )
        combo_reward_text_left = font.render(
            f"Combo Reward: {combo_rewards_left[ind]:0.2f}", True, (255, 255, 255)
        )
        height_pen_text_left = font.render(
            f"Height Penalty: {height_penalties_left[ind]:0.2f}", True, (255, 255, 255)
        )
        hole_pen_text_left = font.render(
            f"Hole Penalty: {hole_penalties_left[ind]:0.2f}", True, (255, 255, 255)
        )
        skyline_pen_text_left = font.render(
            f"Skyline Penalty: {skyline_penalties_left[ind]:0.2f}", True, (255, 255, 255)
        )
        bumpy_pen_text_left = font.render(
            f"Bumpy Penalty: {bumpy_penalties_left[ind]:0.2f}", True, (255, 255, 255)
        )
        death_pen_text_left = font.render(
            f"Death Penalty: {death_penalties_left[ind]:0.2f}", True, (255, 255, 255)
        )

        # Position left player reward texts
        screen.blit(b2b_reward_text_left, (10, base_y))
        screen.blit(combo_reward_text_left, (10, base_y + 20))
        screen.blit(height_pen_text_left, (10, base_y + 40))
        screen.blit(hole_pen_text_left, (10, base_y + 60))
        screen.blit(skyline_pen_text_left, (10, base_y + 80))
        screen.blit(bumpy_pen_text_left, (10, base_y + 100))
        screen.blit(death_pen_text_left, (10, base_y + 120))

        # Right side of left half: Current State Information
        attack_text_left = font.render(
            f"Attack: {int(attacks_left[ind])}", True, (255, 255, 255)
        )
        app_text_left = font.render(f"APP: {apps_left[ind]:0.2f}", True, (255, 255, 255))
        clear_text_left = font.render(
            f"Clear: {int(clears_left[ind])}", True, (255, 255, 255)
        )
        current_b2b_text_left = font.render(
            f"Current B2B: {current_b2b_left[ind]}", True, (255, 255, 255)
        )
        current_combo_text_left = font.render(
            f"Current Combo: {current_combo_left[ind]}", True, (255, 255, 255)
        )
        action_text_left = font.render(
            f"Action: {actions_left[ind]}", True, (255, 255, 255)
        )
        death_text_left = font.render(
            f"Deaths: {sum(deaths_left[: ind + 1])}", True, (255, 255, 255)
        )

        # Position left player state texts
        screen.blit(attack_text_left, (335, base_y))
        screen.blit(app_text_left, (335, base_y + 20))
        screen.blit(clear_text_left, (335, base_y + 40))
        screen.blit(current_b2b_text_left, (335, base_y + 60))
        screen.blit(current_combo_text_left, (335, base_y + 80))
        screen.blit(action_text_left, (335, base_y + 100))
        screen.blit(death_text_left, (335, base_y + 120))

        # RIGHT PLAYER Stats - Replay
        # Left side of right half: Reward Information
        b2b_reward_text_right = font.render(
            f"B2B Reward: {b2b_rewards_right[ind]:0.2f}", True, (255, 255, 255)
        )
        combo_reward_text_right = font.render(
            f"Combo Reward: {combo_rewards_right[ind]:0.2f}", True, (255, 255, 255)
        )
        height_pen_text_right = font.render(
            f"Height Penalty: {height_penalties_right[ind]:0.2f}", True, (255, 255, 255)
        )
        hole_pen_text_right = font.render(
            f"Hole Penalty: {hole_penalties_right[ind]:0.2f}", True, (255, 255, 255)
        )
        skyline_pen_text_right = font.render(
            f"Skyline Penalty: {skyline_penalties_right[ind]:0.2f}", True, (255, 255, 255)
        )
        bumpy_pen_text_right = font.render(
            f"Bumpy Penalty: {bumpy_penalties_right[ind]:0.2f}", True, (255, 255, 255)
        )
        death_pen_text_right = font.render(
            f"Death Penalty: {death_penalties_right[ind]:0.2f}", True, (255, 255, 255)
        )

        # Position right player reward texts
        screen.blit(b2b_reward_text_right, (670, base_y))
        screen.blit(combo_reward_text_right, (670, base_y + 20))
        screen.blit(height_pen_text_right, (670, base_y + 40))
        screen.blit(hole_pen_text_right, (670, base_y + 60))
        screen.blit(skyline_pen_text_right, (670, base_y + 80))
        screen.blit(bumpy_pen_text_right, (670, base_y + 100))
        screen.blit(death_pen_text_right, (670, base_y + 120))

        # Right side of right half: Current State Information
        attack_text_right = font.render(
            f"Attack: {int(attacks_right[ind])}", True, (255, 255, 255)
        )
        app_text_right = font.render(f"APP: {apps_right[ind]:0.2f}", True, (255, 255, 255))
        clear_text_right = font.render(
            f"Clear: {int(clears_right[ind])}", True, (255, 255, 255)
        )
        current_b2b_text_right = font.render(
            f"Current B2B: {current_b2b_right[ind]}", True, (255, 255, 255)
        )
        current_combo_text_right = font.render(
            f"Current Combo: {current_combo_right[ind]}", True, (255, 255, 255)
        )
        action_text_right = font.render(
            f"Action: {actions_right[ind]}", True, (255, 255, 255)
        )
        death_text_right = font.render(
            f"Deaths: {sum(deaths_right[: ind + 1])}", True, (255, 255, 255)
        )

        # Position right player state texts
        screen.blit(attack_text_right, (995, base_y))
        screen.blit(app_text_right, (995, base_y + 20))
        screen.blit(clear_text_right, (995, base_y + 40))
        screen.blit(current_b2b_text_right, (995, base_y + 60))
        screen.blit(current_combo_text_right, (995, base_y + 80))
        screen.blit(action_text_right, (995, base_y + 100))
        screen.blit(death_text_right, (995, base_y + 120))

        pygame.display.update()
