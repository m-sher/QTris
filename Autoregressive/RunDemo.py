import tensorflow as tf
from TetrisModel import PolicyModel
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
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
max_height = 20

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

p_checkpoint = tf.train.Checkpoint(model=p_model)
p_checkpoint_manager = tf.train.CheckpointManager(
    p_checkpoint, "./policy_checkpoints", max_to_keep=3
)
p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()

p_model.build(input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 2), (None, max_len)])

p_model.summary()

py_env = PyTetrisEnv(
    queue_size=queue_size,
    max_holes=max_holes,
    max_height=max_height,
    max_steps=num_steps,
    garbage_chance=0.1,
    garbage_min=1,
    garbage_max=4,
    seed=0,
    idx=0,
)
env = TFPyEnvironment(py_env)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode(
    (670, 800)
)  # Increased height to accommodate text below board
pygame.display.set_caption("Tetris")
font = pygame.font.Font(None, 30)

time_step = env.reset()

frames = []
attacks = []
apps = []
clears = []
actions = []
b2b_rewards = []
combo_rewards = []
current_b2b = []
current_combo = []
height_penalties = []
hole_penalties = []
skyline_penalties = []
bumpy_penalties = []
death_penalties = []

death = 0
running_attacks = 0
running_clears = 0

piece_display = np.load("../PieceDisplay.npy")

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
    board = time_step.observation["board"]
    vis_board = time_step.observation.get("vis_board", None)
    b2b_combo = time_step.observation["b2b_combo"]
    pieces = time_step.observation["pieces"]
    attack = time_step.reward["attack"].numpy()[0]
    clear = time_step.reward["clear"].numpy()[0]
    b2b_reward = time_step.reward["b2b_reward"].numpy()[0]
    combo_reward = time_step.reward["combo_reward"].numpy()[0]
    height_penalty = time_step.reward["height_penalty"].numpy()[0]
    hole_penalty = time_step.reward["hole_penalty"].numpy()[0]
    skyline_penalty = time_step.reward["skyline_penalty"].numpy()[0]
    bumpy_penalty = time_step.reward["bumpy_penalty"].numpy()[0]
    death_penalty = time_step.reward["death_penalty"].numpy()[0]

    # Get current b2b and combo values from environment
    current_b2b_val = py_env._scorer._b2b
    current_combo_val = py_env._scorer._combo

    if time_step.is_last():
        death = t
        running_attacks = 0
        running_clears = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    key_sequence, log_probs, masks, scores = p_model.predict(
        (board, pieces, b2b_combo), greedy=True
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
    dominant_pieces = tf.argmax(
        piece_attention[0], axis=0
    )  # Find dominant piece for each patch (shape: 60)
    dominant_grid = tf.reshape(dominant_pieces, (12, 5))  # Reshape to (12, 5) grid

    # Extract attention intensities for dominant pieces
    dominant_attention = tf.reduce_max(
        piece_attention[0], axis=0
    )  # Get attention value for dominant piece (shape: 60)
    dominant_attention_grid = tf.reshape(
        dominant_attention, (12, 5)
    )  # Reshape to (12, 5) grid

    # Normalize attention intensities for brightness modulation
    attention_min = tf.reduce_min(dominant_attention_grid)
    attention_max = tf.reduce_max(dominant_attention_grid)
    attention_normalized = (dominant_attention_grid - attention_min) / (
        attention_max - attention_min + 1e-8
    )

    # Colorize the scores display based on dominant pieces with intensity modulation
    all_piece_colors = piece_colors[pieces_array]  # Colors for all 7 pieces
    colored_scores = np.zeros((12, 5, 3), dtype=np.uint8)
    dominant_grid_np = dominant_grid.numpy()
    attention_np = attention_normalized.numpy()

    for r in range(12):  # 12 rows
        for c in range(5):  # 5 columns
            piece_idx = dominant_grid_np[r, c]
            intensity = attention_np[r, c]
            colored_scores[r, c] = (all_piece_colors[piece_idx] * intensity).astype(
                np.uint8
            )

    # Get piece display for all 7 pieces
    piece_sidebar = piece_display[pieces_array].reshape((28, 5))

    # Colorize the piece display sidebar
    piece_type_colors = piece_colors[pieces_array]
    colored_sidebar = np.zeros((28, 5, 3), dtype=np.uint8)
    for i in range(7):  # 7 pieces: active, hold, 5 queue
        for r in range(4 * i, 4 * i + 4):  # 4 rows per piece
            for c in range(5):  # 5 columns
                colored_sidebar[r, c] = (
                    piece_type_colors[i] * piece_sidebar[r, c]
                ).astype(np.uint8)

    # Create garbage queue visualization
    garbage_queue = py_env._garbage_queue
    garbage_bar_width = 10  # Thinner width in pixels for the garbage bar
    garbage_bar_height = 24  # Height matches board height
    garbage_surface = np.zeros(
        (garbage_bar_height, garbage_bar_width, 3), dtype=np.uint8
    )

    # Draw garbage sections (bottom to top, with bottom being next to push)
    current_row = garbage_bar_height - 1  # Start from bottom
    for i, (num_rows, empty_column) in enumerate(garbage_queue):
        # Draw red section for this garbage instance
        start_row = max(0, current_row - num_rows + 1)
        for row in range(start_row, current_row + 1):
            if row >= 0 and row < garbage_bar_height:
                garbage_surface[row, :] = [255, 0, 0]  # Red color

        # Add separator line (1 pixel gap) between sections
        if (
            i < len(garbage_queue) - 1 and start_row > 0
        ):  # Not the last section and not at top
            current_row = start_row - 2  # Leave 1 pixel gap (black)
        else:
            current_row = start_row - 1
        if current_row < 0:
            break

    screen.fill((0, 0, 0))

    board_surf = pygame.Surface((10, 24))
    piece_surf = pygame.Surface((5, 28))
    scores_surf = pygame.Surface((5, 12))
    garbage_surf = pygame.Surface((garbage_bar_width, garbage_bar_height))

    if vis_board is not None:
        colored_board = piece_colors[vis_board[0, ..., 0].numpy()]
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
    board_with_border.blit(board_surf, (2, 2))  # Blit board with 2px offset for border

    screen.blit(garbage_surf, (0, 0))  # Garbage bar on the left
    screen.blit(board_with_border, (25, 0))  # Board with border, shifted right less
    screen.blit(piece_surf, (285, 0))  # Piece sidebar adjusted position
    screen.blit(scores_surf, (415, 0))  # Scores adjusted position

    # Add step counter in top left with black background (moved down to avoid slider collision)
    step_text = font.render(f"Step: {t + 1}/{num_steps}", True, (255, 255, 255))
    step_rect = step_text.get_rect()
    step_rect.topleft = (10, 25)
    # Create black background for step counter
    pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))  # Padding around text
    screen.blit(step_text, (10, 25))

    readable_action = "".join(
        [readable_keys.get(k, "") for k in key_sequence.numpy()[0]]
    )

    actions.append(readable_action)

    running_attacks += attack
    running_clears += clear

    attacks.append(running_attacks)
    apps.append(running_attacks / (t - death + 1))
    clears.append(running_clears)
    b2b_rewards.append(b2b_reward)
    combo_rewards.append(combo_reward)
    current_b2b.append(current_b2b_val)
    current_combo.append(current_combo_val)
    height_penalties.append(height_penalty)
    hole_penalties.append(hole_penalty)
    skyline_penalties.append(skyline_penalty)
    bumpy_penalties.append(bumpy_penalty)
    death_penalties.append(death_penalty)

    time_step = env.step(key_sequence)

    # Create black background for text area to prevent overlap
    text_bg_rect = pygame.Rect(0, 610, 670, 190)  # Black background for text area
    pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

    # Draw white dividing line in the middle
    pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

    # Render text below board with white color
    base_y = 615  # Start below the board area

    # LEFT HALF: Reward Information (single column)
    b2b_reward_text = font.render(
        f"B2B Reward: {b2b_reward:0.2f}", True, (255, 255, 255)
    )
    combo_reward_text = font.render(
        f"Combo Reward: {combo_reward:0.2f}", True, (255, 255, 255)
    )
    height_pen_text = font.render(
        f"Height Penalty: {height_penalty:0.2f}", True, (255, 255, 255)
    )
    hole_pen_text = font.render(
        f"Hole Penalty: {hole_penalty:0.2f}", True, (255, 255, 255)
    )
    skyline_pen_text = font.render(
        f"Skyline Penalty: {skyline_penalty:0.2f}", True, (255, 255, 255)
    )
    bumpy_pen_text = font.render(
        f"Bumpy Penalty: {bumpy_penalty:0.2f}", True, (255, 255, 255)
    )
    death_pen_text = font.render(
        f"Death Penalty: {death_penalty:0.2f}", True, (255, 255, 255)
    )

    # Position reward texts in left half (single column)
    screen.blit(b2b_reward_text, (10, base_y))
    screen.blit(combo_reward_text, (10, base_y + 20))
    screen.blit(height_pen_text, (10, base_y + 40))
    screen.blit(hole_pen_text, (10, base_y + 60))
    screen.blit(skyline_pen_text, (10, base_y + 80))
    screen.blit(bumpy_pen_text, (10, base_y + 100))
    screen.blit(death_pen_text, (10, base_y + 120))

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
    action_text = font.render(f"Action: {actions[-1]}", True, (255, 255, 255))

    # Position state texts in right half (single column)
    screen.blit(attack_text, (345, base_y))
    screen.blit(app_text, (345, base_y + 20))
    screen.blit(clear_text, (345, base_y + 40))
    screen.blit(current_b2b_text, (345, base_y + 60))
    screen.blit(current_combo_text, (345, base_y + 80))
    screen.blit(action_text, (345, base_y + 100))

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
    width=585,  # Adjusted width to match new screen size
    height=10,
    min=0,
    max=num_steps - 1,
    step=1,
    colour=(125, 125, 125),
    handleColour=(50, 50, 50),
)

back_btn = Button(
    screen,
    605,  # Adjusted position for new screen width
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
    637,  # Adjusted position for new screen width
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
    text_bg_rect = pygame.Rect(0, 610, 670, 190)  # Black background for text area
    pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

    # Draw white dividing line in the middle
    pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

    # Render text below board for replay with white color
    base_y = 615  # Start below the board area

    # LEFT HALF: Reward Information (single column)
    b2b_reward_text = font.render(
        f"B2B Reward: {b2b_rewards[ind]:0.2f}", True, (255, 255, 255)
    )
    combo_reward_text = font.render(
        f"Combo Reward: {combo_rewards[ind]:0.2f}", True, (255, 255, 255)
    )
    height_pen_text = font.render(
        f"Height Penalty: {height_penalties[ind]:0.2f}", True, (255, 255, 255)
    )
    hole_pen_text = font.render(
        f"Hole Penalty: {hole_penalties[ind]:0.2f}", True, (255, 255, 255)
    )
    skyline_pen_text = font.render(
        f"Skyline Penalty: {skyline_penalties[ind]:0.2f}", True, (255, 255, 255)
    )
    bumpy_pen_text = font.render(
        f"Bumpy Penalty: {bumpy_penalties[ind]:0.2f}", True, (255, 255, 255)
    )
    death_pen_text = font.render(
        f"Death Penalty: {death_penalties[ind]:0.2f}", True, (255, 255, 255)
    )

    # Position reward texts in left half (single column)
    screen.blit(b2b_reward_text, (10, base_y))
    screen.blit(combo_reward_text, (10, base_y + 20))
    screen.blit(height_pen_text, (10, base_y + 40))
    screen.blit(hole_pen_text, (10, base_y + 60))
    screen.blit(skyline_pen_text, (10, base_y + 80))
    screen.blit(bumpy_pen_text, (10, base_y + 100))
    screen.blit(death_pen_text, (10, base_y + 120))

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
    action_text = font.render(f"Action: {actions[ind]}", True, (255, 255, 255))

    # Position state texts in right half (single column)
    screen.blit(attack_text, (345, base_y))
    screen.blit(app_text, (345, base_y + 20))
    screen.blit(clear_text, (345, base_y + 40))
    screen.blit(current_b2b_text, (345, base_y + 60))
    screen.blit(current_combo_text, (345, base_y + 80))
    screen.blit(action_text, (345, base_y + 100))

    pygame.display.update()
