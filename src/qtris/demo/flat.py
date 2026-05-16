import tensorflow as tf
from qtris.models.flat.model import FlatPolicyModel
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
import imageio
import numpy as np
import time

num_envs = 1
piece_dim = 8
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.0
max_len = 15
num_row_tiers = 2
num_sequences = 160 * num_row_tiers

num_steps = 500
queue_size = 5
max_holes = 100
max_height = 18

p_model = FlatPolicyModel(
    batch_size=num_envs,
    piece_dim=piece_dim,
    depth=depth,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
    num_sequences=num_sequences,
)

p_model(
    (
        tf.keras.Input(shape=(24, 10, 1), dtype=tf.float32),
        tf.keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
        tf.keras.Input(shape=(3,), dtype=tf.float32),
    )
)

p_checkpoint = tf.train.Checkpoint(model=p_model)
p_checkpoint_manager = tf.train.CheckpointManager(
    p_checkpoint, "checkpoints/1v1_flat_policy_17k", max_to_keep=3
)
p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()

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
    num_row_tiers=num_row_tiers,
)
env = TFPyEnvironment(py_env)

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
bcg_colors_rgb = np.array(
    [
        [100, 200, 255],  # b2b: cyan
        [255, 255, 100],  # combo: yellow
        [255, 100, 100],  # garbage: red
    ]
)
bcg_labels = ["B2B", "Combo", "Garb"]

time_step = env.reset()

frames = []
attacks = []
apps = []
clears = []
actions = []
attack_rewards = []
total_rewards = []
current_b2b = []
current_combo = []
current_garbage = []

death = 0
running_attacks = 0
running_clears = 0

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
    board = time_step.observation["board"]
    vis_board = time_step.observation.get("vis_board", None)
    b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
    pieces = time_step.observation["pieces"]
    valid_sequences = time_step.observation["sequences"]
    attack = time_step.reward["attack"].numpy()[0]
    clear = time_step.reward["clear"].numpy()[0]
    attack_reward = time_step.reward["attack_reward"].numpy()[0]
    total_reward = time_step.reward["total_reward"].numpy()[0]

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

    key_sequence, log_prob, action_index, scores = p_model.predict(
        (board, pieces, b2b_combo_garbage),
        greedy=True,
        valid_sequences=valid_sequences,
        temperature=1.0,
    )

    pieces_array = pieces.numpy()
    if pieces_array.ndim > 1:
        pieces_array = pieces_array[0]

    piece_attention = tf.reduce_sum(scores, axis=[0, 2])
    # Slice out BCG tokens: keep only piece queries (:7) and patch keys (:60)
    piece_patch_attn = piece_attention[0, :7, :60]
    dominant_pieces = tf.argmax(piece_patch_attn, axis=0)
    dominant_grid = tf.reshape(dominant_pieces, (12, 5))

    dominant_attention = tf.reduce_max(piece_patch_attn, axis=0)
    dominant_attention_grid = tf.reshape(dominant_attention, (12, 5))

    # BCG (b2b, combo, garbage) attention over the 60 board patches
    bcg_patch_attn = piece_attention[0, 7:10, :60].numpy()  # (3, 60)
    bcg_grids = bcg_patch_attn.reshape(3, 12, 5)
    bcg_colored_heatmaps = np.zeros((3, 12, 5, 3), dtype=np.uint8)
    for i in range(3):
        g = bcg_grids[i]
        g_min, g_max = g.min(), g.max()
        norm = (g - g_min) / (g_max - g_min + 1e-8)
        bcg_colored_heatmaps[i] = (
            bcg_colors_rgb[i][None, None] * norm[..., None]
        ).astype(np.uint8)

    attention_min = tf.reduce_min(dominant_attention_grid)
    attention_max = tf.reduce_max(dominant_attention_grid)
    attention_normalized = (dominant_attention_grid - attention_min) / (
        attention_max - attention_min + 1e-8
    )

    all_piece_colors = piece_colors[pieces_array]
    colored_scores = np.zeros((12, 5, 3), dtype=np.uint8)
    dominant_grid_np = dominant_grid.numpy()
    attention_np = attention_normalized.numpy()

    for r in range(12):
        for c in range(5):
            piece_idx = dominant_grid_np[r, c]
            intensity = attention_np[r, c]
            colored_scores[r, c] = (all_piece_colors[piece_idx] * intensity).astype(
                np.uint8
            )

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
    for i, (num_rows, empty_column, timing) in enumerate(garbage_queue):
        start_row = max(0, current_row - num_rows + 1)
        for row in range(start_row, current_row + 1):
            if row >= 0 and row < garbage_bar_height:
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

    # BCG attention panel (b2b, combo, garbage attention over board patches)
    bcg_vals = [current_b2b_val, current_combo_val, current_garbage_val]
    for i in range(3):
        hx = bcg_panel_x + i * (bcg_heatmap_w + bcg_gap)
        label_text = small_font.render(
            f"{bcg_labels[i]}: {bcg_vals[i]}", True, (255, 255, 255)
        )
        screen.blit(label_text, (hx, bcg_label_y))
        bcg_surf = pygame.Surface((5, 12))
        pygame.surfarray.blit_array(
            bcg_surf, bcg_colored_heatmaps[i].transpose(1, 0, 2)
        )
        bcg_surf = pygame.transform.scale(bcg_surf, (bcg_heatmap_w, bcg_heatmap_h))
        screen.blit(bcg_surf, (hx, bcg_heatmap_y))

    step_text = font.render(f"Step: {t + 1}/{num_steps}", True, (255, 255, 255))
    step_rect = step_text.get_rect()
    step_rect.topleft = (10, 25)
    pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))
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
    attack_rewards.append(attack_reward)
    total_rewards.append(total_reward)
    current_b2b.append(current_b2b_val)
    current_combo.append(current_combo_val)
    current_garbage.append(current_garbage_val)

    time_step = env.step(key_sequence)

    text_bg_rect = pygame.Rect(0, 610, screen_w, 190)
    pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

    pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

    base_y = 615

    attack_reward_text = font.render(
        f"Attack Reward: {attack_reward:0.2f}", True, (255, 255, 255)
    )
    total_reward_text = font.render(
        f"Total Reward: {total_reward:0.2f}", True, (255, 255, 255)
    )

    screen.blit(attack_reward_text, (10, base_y))
    screen.blit(total_reward_text, (10, base_y + 20))

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
    writer = imageio.get_writer("DemoFlat.mp4", fps=30)
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

paused = True


def toggle_pause():
    global paused
    paused = not paused
    play_btn.setText("Play" if paused else "Pause")


play_btn = Button(
    screen,
    605,
    25,
    60,
    20,
    text="Play",
    fontSize=16,
    margin=0,
    onClick=toggle_pause,
)

speed_slider = Slider(
    screen,
    x=10,
    y=60,
    width=200,
    height=10,
    min=1,
    max=60,
    step=1,
    initial=30,
    colour=(125, 125, 125),
    handleColour=(50, 50, 50),
)

last_step_time = pygame.time.get_ticks()

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not paused:
        current_time = pygame.time.get_ticks()
        delay = int(1000 / speed_slider.getValue())
        if current_time - last_step_time >= delay:
            current_val = slider.getValue()
            if current_val < num_steps - 1:
                slider.setValue(current_val + 1)
                last_step_time = current_time
            else:
                paused = True
                play_btn.setText("Play")

    screen.fill((0, 0, 0))

    speed_text = font.render(
        f"Speed: {speed_slider.getValue()} FPS", True, (255, 255, 255)
    )
    screen.blit(speed_text, (220, 55))

    ind = slider.getValue()
    frame = frames[ind]

    pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))

    step_text = font.render(f"Step: {ind + 1}/{num_steps}", True, (255, 255, 255))
    step_rect = step_text.get_rect()
    step_rect.topleft = (10, 25)
    pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 4))
    screen.blit(step_text, (10, 25))

    pygame_widgets.update(events)

    text_bg_rect = pygame.Rect(0, 610, screen_w, 190)
    pygame.draw.rect(screen, (0, 0, 0), text_bg_rect)

    pygame.draw.line(screen, (255, 255, 255), (335, 610), (335, 800), 2)

    base_y = 615

    attack_reward_text = font.render(
        f"Attack Reward: {attack_rewards[ind]:0.2f}", True, (255, 255, 255)
    )
    total_reward_text = font.render(
        f"Total Reward: {total_rewards[ind]:0.2f}", True, (255, 255, 255)
    )

    screen.blit(attack_reward_text, (10, base_y))
    screen.blit(total_reward_text, (10, base_y + 20))

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
