import importlib
import time

import numpy as np
import pygame
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisModel import PolicyModel

import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
import imageio

# Model params
piece_dim = 8
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.05
output_dim = 160

# Environment params
num_steps = 500
queue_size = 5
max_holes = 50
max_height = 18
max_len = 15
garbage_chance = 0.1
garbage_min = 1
garbage_max = 4
seed = 0

p_model = PolicyModel(
    piece_dim=piece_dim,
    depth=depth,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
    output_dim=output_dim,
)

p_checkpoint = tf.train.Checkpoint(model=p_model)
p_checkpoint_manager = tf.train.CheckpointManager(
    p_checkpoint, "./policy_checkpoints", max_to_keep=3
)
p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()

p_model.build(
    input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 2)]
)
p_model.summary()

py_env = PyTetrisEnv(
    queue_size=queue_size,
    max_holes=max_holes,
    max_height=max_height,
    max_steps=num_steps,
    max_len=max_len,
    garbage_chance=garbage_chance,
    garbage_min=garbage_min,
    garbage_max=garbage_max,
    seed=seed,
    idx=0,
)
env = TFPyEnvironment(py_env)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((670, 800))
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
    Keys.HOLD: "h",
    Keys.TAP_LEFT: "l",
    Keys.TAP_RIGHT: "r",
    Keys.DAS_LEFT: "L",
    Keys.DAS_RIGHT: "R",
    Keys.CLOCKWISE: "c",
    Keys.ANTICLOCKWISE: "a",
    Keys.ROTATE_180: "1",
    Keys.SOFT_DROP: "s",
    Keys.HARD_DROP: "H",
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
    ],
    dtype=np.uint8,
)

start = time.time()
for t in range(num_steps):
    observation = time_step.observation
    board = observation["board"]
    vis_board = observation.get("vis_board", None)
    b2b_combo = observation["b2b_combo"]
    pieces = observation["pieces"]
    sequences = observation["sequences"]
    action_mask = tf.reduce_any(sequences != Keys.PAD, axis=-1)

    reward = time_step.reward
    attack = float(reward["attack"].numpy()[0])
    clear = float(reward["clear"].numpy()[0])
    b2b_reward = float(reward["b2b_reward"].numpy()[0])
    combo_reward = float(reward["combo_reward"].numpy()[0])
    height_penalty = float(reward["height_penalty"].numpy()[0])
    hole_penalty = float(reward["hole_penalty"].numpy()[0])
    skyline_penalty = float(reward["skyline_penalty"].numpy()[0])
    bumpy_penalty = float(reward["bumpy_penalty"].numpy()[0])
    death_penalty = float(reward["death_penalty"].numpy()[0])

    current_b2b_val = py_env._scorer._b2b
    current_combo_val = py_env._scorer._combo

    if time_step.is_last():
        death = t
        running_attacks = 0
        running_clears = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action, log_prob, scores = p_model.predict(
        (board, pieces, b2b_combo, action_mask), greedy=True
    )

    key_sequence = tf.gather(sequences, action, batch_dims=1)

    pieces_array = pieces.numpy()
    if pieces_array.ndim > 1:
        pieces_array = pieces_array[0]

    scores_tensor = tf.stack(scores, axis=0)
    piece_attention = tf.reduce_sum(scores_tensor, axis=[0, 2])
    dominant_pieces = tf.argmax(piece_attention[0], axis=0)
    dominant_grid = tf.reshape(dominant_pieces, (12, 5))
    dominant_attention = tf.reduce_max(piece_attention[0], axis=0)
    dominant_attention_grid = tf.reshape(dominant_attention, (12, 5))
    attention_min = tf.reduce_min(dominant_attention_grid)
    attention_max = tf.reduce_max(dominant_attention_grid)
    attention_range = attention_max - attention_min + 1e-8
    attention_normalized = (dominant_attention_grid - attention_min) / attention_range

    all_piece_colors = piece_colors[pieces_array]
    colored_scores = np.zeros((12, 5, 3), dtype=np.uint8)
    dominant_grid_np = dominant_grid.numpy()
    attention_np = attention_normalized.numpy()
    for r in range(12):
        for c in range(5):
            piece_idx = dominant_grid_np[r, c]
            intensity = attention_np[r, c]
            colored_scores[r, c] = (
                all_piece_colors[piece_idx] * intensity
            ).astype(np.uint8)

    piece_sidebar = piece_display[pieces_array].reshape((28, 5))
    piece_type_colors = piece_colors[pieces_array]
    colored_sidebar = np.zeros((28, 5, 3), dtype=np.uint8)
    for i in range(7):
        for r in range(4 * i, 4 * i + 4):
            for c in range(5):
                colored_sidebar[r, c] = (
                    piece_type_colors[i] * piece_sidebar[r, c]
                ).astype(np.uint8)

    garbage_queue = py_env._garbage_queue
    garbage_bar_width = 10
    garbage_bar_height = 24
    garbage_surface = np.zeros(
        (garbage_bar_height, garbage_bar_width, 3), dtype=np.uint8
    )
    current_row = garbage_bar_height - 1
    for i, (num_rows, empty_column) in enumerate(garbage_queue):
        start_row = max(0, current_row - num_rows + 1)
        for row in range(start_row, current_row + 1):
            if 0 <= row < garbage_bar_height:
                garbage_surface[row, :] = [255, 0, 0]
        if i < len(garbage_queue) - 1 and start_row > 0:
            current_row = start_row - 2
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
    garbage_surf = pygame.transform.scale(garbage_surf, (25, 600))

    board_with_border = pygame.Surface((254, 604))
    board_with_border.fill((255, 255, 255))
    board_with_border.blit(board_surf, (2, 2))

    screen.blit(garbage_surf, (0, 0))
    screen.blit(board_with_border, (25, 0))
    screen.blit(piece_surf, (285, 0))
    screen.blit(scores_surf, (415, 0))

    step_text = font.render(f"Step: {t + 1}/{num_steps}", True, (255, 255, 255))
    step_rect = step_text.get_rect()
    step_rect.topleft = (10, 25)
    pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))
    screen.blit(step_text, (10, 25))

    sequence_tokens = key_sequence.numpy()[0]
    readable_action = "".join(
        readable_keys.get(int(k), "")
        for k in sequence_tokens
        if k not in (Keys.START, Keys.PAD)
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

    text_bg_rect = pygame.Rect(0, 610, 670, 190)
    pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)
    pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

    base_y = 615
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

    screen.blit(b2b_reward_text, (10, base_y))
    screen.blit(combo_reward_text, (10, base_y + 20))
    screen.blit(height_pen_text, (10, base_y + 40))
    screen.blit(hole_pen_text, (10, base_y + 60))
    screen.blit(skyline_pen_text, (10, base_y + 80))
    screen.blit(bumpy_pen_text, (10, base_y + 100))
    screen.blit(death_pen_text, (10, base_y + 120))

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
    width=585,
    height=10,
    min=0,
    max=num_steps - 1,
    step=1,
    colour=(125, 125, 125),
    handleColour=(50, 50, 50),
)

back_btn = Button(
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

fwd_btn = Button(
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

    step_text = font.render(f"Step: {ind + 1}/{num_steps}", True, (255, 255, 255))
    step_rect = step_text.get_rect()
    step_rect.topleft = (10, 25)
    pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))
    screen.blit(step_text, (10, 25))

    pygame_widgets.update(events)

    text_bg_rect = pygame.Rect(0, 610, 670, 190)
    pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)
    pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

    base_y = 615
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

    screen.blit(b2b_reward_text, (10, base_y))
    screen.blit(combo_reward_text, (10, base_y + 20))
    screen.blit(height_pen_text, (10, base_y + 40))
    screen.blit(hole_pen_text, (10, base_y + 60))
    screen.blit(skyline_pen_text, (10, base_y + 80))
    screen.blit(bumpy_pen_text, (10, base_y + 100))
    screen.blit(death_pen_text, (10, base_y + 120))

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

    screen.blit(attack_text, (345, base_y))
    screen.blit(app_text, (345, base_y + 20))
    screen.blit(clear_text, (345, base_y + 40))
    screen.blit(current_b2b_text, (345, base_y + 60))
    screen.blit(current_combo_text, (345, base_y + 80))
    screen.blit(action_text, (345, base_y + 100))

    pygame.display.update()
