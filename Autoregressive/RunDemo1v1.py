import argparse
import tensorflow as tf
from TetrisModel import PolicyModel
from TetrisEnv.PyTetris1v1Env import PyTetris1v1Env
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
import imageio
import numpy as np
import time


# Model params
piece_dim = 8
key_dim = 12
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.1
max_len = 15
queue_size = 5
num_row_tiers = 2

piece_colors = np.array([
    [0, 0, 0],
    [0, 255, 255],
    [0, 0, 255],
    [255, 127, 0],
    [255, 200, 0],
    [0, 255, 0],
    [255, 0, 255],
    [255, 0, 0],
    [127, 127, 127],
])

readable_keys = {
    1: "h", 2: "l", 3: "r", 4: "L", 5: "R",
    6: "c", 7: "a", 8: "1", 9: "s", 10: "H",
}


def load_policy(checkpoint_dir):
    model = PolicyModel(
        batch_size=1,
        piece_dim=piece_dim,
        key_dim=key_dim,
        depth=depth,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=key_dim,
    )
    model.build(
        input_shape=[(None, 24, 10, 1), (None, queue_size + 2), (None, 3), (None, max_len)]
    )
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Loaded checkpoint from {checkpoint_dir}", flush=True)
    else:
        print(f"No checkpoint found in {checkpoint_dir}, using random weights", flush=True)
    return model


def draw_garbage_bar(env_instance, height=24, width=4):
    """Create a garbage queue visualization array (height, width, 3)."""
    surface = np.zeros((height, width, 3), dtype=np.uint8)
    current_row = height - 1
    for i, (num_rows, _) in enumerate(env_instance._garbage_queue):
        start_row = max(0, current_row - num_rows + 1)
        for row in range(start_row, current_row + 1):
            if 0 <= row < height:
                surface[row, :] = [255, 0, 0]
        if i < len(env_instance._garbage_queue) - 1 and start_row > 0:
            current_row = start_row - 2
        else:
            current_row = start_row - 1
        if current_row < 0:
            break
    return surface


def main():
    parser = argparse.ArgumentParser(description="1v1 Tetris Demo")
    parser.add_argument("--p1", required=True, help="P1 checkpoint directory")
    parser.add_argument("--p2", required=True, help="P2 checkpoint directory")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per game")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true", help="Use greedy action selection")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    num_steps = args.steps

    # Load models
    p1_model = load_policy(args.p1)
    p2_model = load_policy(args.p2)
    p1_model.summary()

    # Create environment
    py_env = PyTetris1v1Env(
        queue_size=queue_size,
        max_holes=50,
        max_height=18,
        max_steps=num_steps,
        max_len=max_len,
        pathfinding=True,
        seed=args.seed,
        idx=0,
        num_row_tiers=num_row_tiers,
    )
    env = TFPyEnvironment(py_env)

    # Layout constants
    board_w, board_h = 250, 600
    garbage_w = 20
    gap = 30
    stats_h = 120
    margin = 10

    panel_w = garbage_w + board_w
    screen_w = margin + panel_w + gap + panel_w + margin
    screen_h = board_h + stats_h + 30  # 30 for top bar

    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tetris 1v1")
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 36)

    time_step = env.reset()

    frames = []
    p1_stats = {"attacks": [], "clears": [], "b2b": [], "combo": [], "garbage": [], "app": []}
    p2_stats = {"attacks": [], "clears": [], "b2b": [], "combo": [], "garbage": [], "app": []}
    p1_running_attacks = 0
    p2_running_attacks = 0
    p1_running_clears = 0
    p2_running_clears = 0
    p1_death_step = 0
    p2_death_step = 0
    winner = None

    start = time.time()
    for t in range(num_steps):
        # P1 observations
        board1 = time_step.observation["board"]
        pieces1 = time_step.observation["pieces"]
        bcg1 = time_step.observation["b2b_combo_garbage"]
        seqs1 = time_step.observation["sequences"]

        # P2 observations
        board2 = time_step.observation["opp_board"]
        pieces2 = time_step.observation["opp_pieces"]
        bcg2 = time_step.observation["opp_b2b_combo_garbage"]
        seqs2 = time_step.observation["opp_sequences"]

        # Generate actions
        p1_keys, _, _, _ = p1_model.predict(
            (board1, pieces1, bcg1),
            greedy=args.greedy, valid_sequences=seqs1, temperature=args.temperature,
        )
        p2_keys, _, _, _ = p2_model.predict(
            (board2, pieces2, bcg2),
            greedy=args.greedy, valid_sequences=seqs2, temperature=args.temperature,
        )

        combined = tf.concat([p1_keys, p2_keys], axis=-1)
        time_step = env.step(combined)

        # Collect stats from underlying envs
        attack1 = time_step.reward["attack"].numpy()[0]
        clear1 = time_step.reward["clear"].numpy()[0]

        p1_running_attacks += attack1
        p1_running_clears += clear1
        p1_stats["attacks"].append(p1_running_attacks)
        p1_stats["clears"].append(p1_running_clears)
        p1_stats["b2b"].append(py_env._env1._scorer._b2b)
        p1_stats["combo"].append(py_env._env1._scorer._combo)
        p1_stats["garbage"].append(py_env._env1._get_total_garbage())
        p1_stats["app"].append(p1_running_attacks / (t - p1_death_step + 1))

        attack2 = time_step.reward["opp_attack"].numpy()[0]
        clear2 = time_step.reward["opp_clear"].numpy()[0]
        p2_running_attacks += attack2
        p2_running_clears += clear2
        p2_stats["attacks"].append(p2_running_attacks)
        p2_stats["clears"].append(p2_running_clears)
        p2_stats["b2b"].append(py_env._env2._scorer._b2b)
        p2_stats["combo"].append(py_env._env2._scorer._combo)
        p2_stats["garbage"].append(py_env._env2._get_total_garbage())
        p2_stats["app"].append(p2_running_attacks / (t - p2_death_step + 1))

        # --- Render ---
        screen.fill((0, 0, 0))

        # Step counter
        step_text = font.render(f"Step {t + 1}/{num_steps}", True, (255, 255, 255))
        screen.blit(step_text, (screen_w // 2 - step_text.get_width() // 2, 5))

        top_y = 28

        # P1 board (left)
        p1_x = margin
        vis1 = py_env._env1._vis_board
        colored1 = piece_colors[vis1]
        board1_surf = pygame.Surface((10, 24))
        pygame.surfarray.blit_array(board1_surf, colored1.transpose(1, 0, 2))
        board1_surf = pygame.transform.scale(board1_surf, (board_w, board_h))

        # P1 garbage bar
        g1 = draw_garbage_bar(py_env._env1)
        g1_surf = pygame.Surface((g1.shape[1], g1.shape[0]))
        pygame.surfarray.blit_array(g1_surf, g1.transpose(1, 0, 2))
        g1_surf = pygame.transform.scale(g1_surf, (garbage_w, board_h))

        screen.blit(g1_surf, (p1_x, top_y))
        screen.blit(board1_surf, (p1_x + garbage_w, top_y))

        # P1 label
        p1_label = big_font.render("P1", True, (100, 200, 255))
        screen.blit(p1_label, (p1_x + garbage_w + 5, top_y + 5))

        # P2 board (right)
        p2_x = margin + panel_w + gap
        vis2 = py_env._env2._vis_board
        colored2 = piece_colors[vis2]
        board2_surf = pygame.Surface((10, 24))
        pygame.surfarray.blit_array(board2_surf, colored2.transpose(1, 0, 2))
        board2_surf = pygame.transform.scale(board2_surf, (board_w, board_h))

        # P2 garbage bar
        g2 = draw_garbage_bar(py_env._env2)
        g2_surf = pygame.Surface((g2.shape[1], g2.shape[0]))
        pygame.surfarray.blit_array(g2_surf, g2.transpose(1, 0, 2))

        screen.blit(board2_surf, (p2_x, top_y))
        screen.blit(g2_surf, (p2_x + board_w, top_y))

        # P2 label
        p2_label = big_font.render("P2", True, (255, 150, 100))
        screen.blit(p2_label, (p2_x + 5, top_y + 5))

        # Stats area
        stats_y = top_y + board_h + 5

        def draw_stats(x, stats_dict, idx, color):
            lines = [
                f"ATK: {int(stats_dict['attacks'][idx])}  APP: {stats_dict['app'][idx]:.2f}",
                f"Clears: {int(stats_dict['clears'][idx])}",
                f"B2B: {stats_dict['b2b'][idx]}  Combo: {stats_dict['combo'][idx]}",
                f"Garbage: {stats_dict['garbage'][idx]}",
            ]
            for i, line in enumerate(lines):
                text = font.render(line, True, color)
                screen.blit(text, (x, stats_y + i * 22))

        draw_stats(p1_x + garbage_w, p1_stats, t, (100, 200, 255))
        draw_stats(p2_x, p2_stats, t, (255, 150, 100))

        # Check game end
        if time_step.is_last() and winner is None:
            win_reward = time_step.reward["win"].numpy()[0]
            if win_reward > 0:
                winner = "P1 WINS"
            else:
                winner = "P2 WINS"

        if winner:
            win_text = big_font.render(winner, True, (255, 255, 0))
            screen.blit(win_text, (screen_w // 2 - win_text.get_width() // 2, top_y + board_h // 2))

        pygame.display.update()
        frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if time_step.is_last():
            break

    elapsed = time.time() - start
    actual_steps = t + 1
    print(f"Time: {elapsed:.2f}s | Steps: {actual_steps} | {elapsed/actual_steps:.3f}s/step")
    print(f"Result: {winner or 'Timeout'}")

    if input("Save video? (y/n) ").lower() == "y":
        writer = imageio.get_writer("Demo1v1.mp4", fps=30)
        for frame in frames:
            for _ in range(30 // 5):
                writer.append_data(frame)
        writer.close()
        print("Saved Demo1v1.mp4")

    # Replay slider
    slider = Slider(
        screen, x=10, y=5, width=screen_w - 80, height=10,
        min=0, max=len(frames) - 1, step=1,
        colour=(125, 125, 125), handleColour=(50, 50, 50),
    )
    back_btn = Button(
        screen, screen_w - 60, 0, 28, 20, text="<", fontSize=16, margin=0,
        onClick=lambda: slider.setValue(max(0, slider.getValue() - 1)),
    )
    fwd_btn = Button(
        screen, screen_w - 28, 0, 28, 20, text=">", fontSize=16, margin=0,
        onClick=lambda: slider.setValue(min(len(frames) - 1, slider.getValue() + 1)),
    )

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        ind = slider.getValue()
        pygame.surfarray.blit_array(screen, frames[ind].swapaxes(0, 1))
        pygame_widgets.update(events)

        step_text = font.render(f"Step {ind + 1}/{len(frames)}", True, (255, 255, 255))
        bg_rect = step_text.get_rect(topleft=(10, 20))
        pygame.draw.rect(screen, (0, 0, 0), bg_rect.inflate(10, 4))
        screen.blit(step_text, (10, 20))

        pygame.display.update()


if __name__ == "__main__":
    main()