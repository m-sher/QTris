"""1v1 demo for the placement AlphaZero model.

Two placement nets duel in PyTetris1v1Env (real garbage exchange). Each move, per-player PUCT
MCTS picks a placement; its key sequence is reconstructed from the env pathfinder and stepped
through the full env (so the colored board renders, same path the solo/AR demos use). Renders
both boards side by side and saves an mp4. Run via `demo placement --mode 1v1 --checkpoint P1
--opponent P2` (same path for self-play).
"""

import time

import numpy as np
import pygame
import pygame_widgets
import tensorflow as tf
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider
from gymnasium.vector import SyncVectorEnv
from TetrisEnv.tf_vec_env import TFVecEnv

from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetris1v1Env import PyTetris1v1Env
from qtris.demo.constants import PIECE_COLORS
from qtris.demo.rendering import colorize_piece_sidebar, draw_garbage_bar
from qtris.demo.utils import load_checkpoint, load_piece_display, save_frames_as_video
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.training._1v1_placement_az import _build_net

# Model params (match the 1v1 AZ trainer)
piece_dim = 8
depth = 64
num_heads = 4
num_layers = 4
queue_size = 5
max_len = 15
num_row_tiers = 2


def load_net(checkpoint_dir):
    net = _build_net(1, piece_dim, depth, num_heads, num_layers, queue_size)
    load_checkpoint(net, checkpoint_dir)
    return net


def main(cli_args):
    p1_ckpt = str(cli_args.checkpoint)
    p2_ckpt = str(cli_args.opponent)
    sims = (
        getattr(cli_args, "num_simulations", 0) or 256
    )  # 1v1 is MCTS-only; 0 -> default budget
    cpuct = getattr(cli_args, "c_puct", 1.5)
    leaves = getattr(cli_args, "leaves_per_round", 4)
    num_steps = getattr(cli_args, "max_game_steps", 500)
    seed = getattr(cli_args, "seed", 0)

    if (
        getattr(cli_args, "garbage_traces", None)
        or getattr(cli_args, "garbage_chance", 0.15) != 0.15
    ):
        print(
            "1v1: garbage comes from the opponent; --garbage-* flags ignored.",
            flush=True,
        )

    p1_net = load_net(p1_ckpt)
    p2_net = load_net(p2_ckpt)
    p1_net.summary()

    cfg = MCTSConfig(
        num_simulations=sims,
        c_puct=cpuct,
        dirichlet_eps=0.0,
        leaves_per_round=leaves,
        gamma=1.0,
        w_attack=0.05,
        w_death=1.0,
        w_b2b=0.06,
    )
    mcts1 = PlacementMCTS(p1_net, cfg)
    mcts2 = PlacementMCTS(p2_net, cfg)

    py_env = PyTetris1v1Env(
        queue_size=queue_size,
        max_holes=50,
        max_steps=num_steps,
        max_len=max_len,
        pathfinding=True,
        seed=seed,
        idx=0,
        num_row_tiers=num_row_tiers,
    )
    env = TFVecEnv(SyncVectorEnv([lambda: py_env]))
    piece_display = load_piece_display()

    # Layout: [garbage | board | queue sidebar] per side, sidebars facing the center gap.
    board_w, board_h = 250, 600
    garbage_w = 25
    sidebar_w = 125  # solo demo scales the (5, 28) piece sidebar to 125 x 600
    gap = 30
    stats_h = 120
    margin = 10
    panel_w = garbage_w + board_w + sidebar_w
    screen_w = margin + panel_w + gap + panel_w + margin
    screen_h = 28 + board_h + 5 + stats_h + margin

    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tetris 1v1 (placement AZ)")
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 36)

    time_step = env.reset()

    frames = []
    p1 = {"atk": 0.0, "b2b": 0, "combo": 0, "garbage": 0, "app": 0.0, "value": 0.0}
    p2 = {"atk": 0.0, "b2b": 0, "combo": 0, "garbage": 0, "app": 0.0, "value": 0.0}
    p1_atk = p2_atk = 0.0
    winner = None

    def mcts_keys(sub_env, mcts, side):
        """MCTS-chosen placement -> key sequence (via the env pathfinder), like the solo demo."""
        res = mcts.search([sub_env], 1.0, 0.0)[0]
        if res["dead"]:
            forced = np.full(max_len, Keys.PAD, dtype=np.int64)
            forced[0], forced[1] = Keys.START, Keys.HARD_DROP
            return tf.constant(forced[None], dtype=tf.int64)
        side["value"] = res["value"]
        is_hold, rot, norm_col, _landing, spin = res["descriptor"]
        action_index = is_hold * 160 + rot * 40 + norm_col * 4 + spin
        _, _, cand_seqs = sub_env._enumerate_placement_candidates()
        return tf.constant(cand_seqs[action_index][None], dtype=tf.int64)

    def blit_scaled(rgb, x, y, w, h):
        surf = pygame.Surface((rgb.shape[1], rgb.shape[0]))
        pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surf, (w, h)), (x, y))

    def queue_rgb(sub_env):
        pieces = np.array(
            [sub_env._active_piece.piece_type.value, sub_env._hold_piece.value]
            + [p.value for p in sub_env._queue],
            dtype=np.int64,
        )[: queue_size + 2]
        return colorize_piece_sidebar(piece_display, pieces, PIECE_COLORS)

    def draw_side(sub_env, x, mirror, label, color):
        g = draw_garbage_bar(sub_env)
        b = PIECE_COLORS[
            sub_env._vis_board[-24:]
        ]  # last 24 rows = visible board (typed)
        s = queue_rgb(sub_env)
        if mirror:  # P2: sidebar | board | garbage (sidebar toward the center gap)
            blit_scaled(s, x, 28, sidebar_w, board_h)
            blit_scaled(b, x + sidebar_w, 28, board_w, board_h)
            blit_scaled(g, x + sidebar_w + board_w, 28, garbage_w, board_h)
            label_x = x + sidebar_w + 5
        else:  # P1: garbage | board | sidebar
            blit_scaled(g, x, 28, garbage_w, board_h)
            blit_scaled(b, x + garbage_w, 28, board_w, board_h)
            blit_scaled(s, x + garbage_w + board_w, 28, sidebar_w, board_h)
            label_x = x + garbage_w + 5
        screen.blit(big_font.render(label, True, color), (label_x, 33))

    def draw_stats(x, s, color):
        lines = [
            f"ATK: {int(s['atk'])}  APP: {s['app']:.2f}",
            f"B2B: {s['b2b']}  Combo: {s['combo']}",
            f"Garbage: {s['garbage']}",
            f"Value: {s['value']:+.2f}",
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, color), (x, 28 + board_h + 5 + i * 22))

    def render(t):
        screen.fill((0, 0, 0))
        step_text = font.render(f"Step {t + 1}/{num_steps}", True, (255, 255, 255))
        screen.blit(step_text, (screen_w // 2 - step_text.get_width() // 2, 5))
        p1_x = margin
        p2_x = margin + panel_w + gap
        draw_side(py_env._env1, p1_x, False, "P1", (100, 200, 255))
        draw_side(py_env._env2, p2_x, True, "P2", (255, 150, 100))
        draw_stats(p1_x + garbage_w, p1, (100, 200, 255))
        draw_stats(p2_x + sidebar_w, p2, (255, 150, 100))
        if winner:
            win_text = big_font.render(winner, True, (255, 255, 0))
            screen.blit(
                win_text, (screen_w // 2 - win_text.get_width() // 2, 28 + board_h // 2)
            )
        pygame.display.update()
        frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

    start = time.time()
    t = 0
    for t in range(num_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        p1_keys = mcts_keys(py_env._env1, mcts1, p1)
        p2_keys = mcts_keys(py_env._env2, mcts2, p2)
        time_step = env.step(tf.concat([p1_keys, p2_keys], axis=-1))

        p1_atk += time_step.reward["attack"].numpy()[0]
        p2_atk += time_step.reward["opp_attack"].numpy()[0]
        p1["atk"], p1["app"] = p1_atk, p1_atk / (t + 1)
        p2["atk"], p2["app"] = p2_atk, p2_atk / (t + 1)
        p1["b2b"], p1["combo"], p1["garbage"] = (
            py_env._env1._scorer._b2b,
            py_env._env1._scorer._combo,
            py_env._env1._get_total_garbage(),
        )
        p2["b2b"], p2["combo"], p2["garbage"] = (
            py_env._env2._scorer._b2b,
            py_env._env2._scorer._combo,
            py_env._env2._get_total_garbage(),
        )

        if time_step.done and winner is None:
            win_reward = time_step.reward["win"].numpy()[0]
            if win_reward > 0:
                winner = "P1 WINS"
            elif t + 1 >= num_steps:
                winner = "DRAW (timeout)"
            else:
                winner = "P2 WINS"

        render(t)
        if time_step.done:
            break

    elapsed = time.time() - start
    actual = t + 1
    print(f"Time: {elapsed:.2f}s | Steps: {actual} | {elapsed / actual:.3f}s/step")
    print(f"Result: {winner or 'Timeout'}")

    save_frames_as_video(frames, "DemoPlacement1v1.mp4")

    # Replay slider (mirrors ar_1v1)
    slider = Slider(
        screen,
        x=10,
        y=5,
        width=screen_w - 280,
        height=10,
        min=0,
        max=len(frames) - 1,
        step=1,
        colour=(125, 125, 125),
        handleColour=(50, 50, 50),
    )
    # Held in vars so pygame_widgets' WeakSet doesn't GC them (bare exprs vanish).
    _back_btn = Button(
        screen,
        screen_w - 60,
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
        screen_w - 28,
        0,
        28,
        20,
        text=">",
        fontSize=16,
        margin=0,
        onClick=lambda: slider.setValue(min(len(frames) - 1, slider.getValue() + 1)),
    )

    paused = True

    def toggle_pause():
        nonlocal paused
        paused = not paused
        play_btn.setText("Play" if paused else "Pause")

    play_btn = Button(
        screen,
        screen_w - 150,
        0,
        60,
        20,
        text="Play",
        fontSize=16,
        margin=0,
        onClick=toggle_pause,
    )

    speed_slider = Slider(
        screen,
        x=screen_w - 260,
        y=5,
        width=100,
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
                return

        if not paused:
            current_time = pygame.time.get_ticks()
            delay = int(1000 / speed_slider.getValue())
            if current_time - last_step_time >= delay:
                current_val = slider.getValue()
                if current_val < len(frames) - 1:
                    slider.setValue(current_val + 1)
                    last_step_time = current_time
                else:
                    paused = True
                    play_btn.setText("Play")

        ind = slider.getValue()
        pygame.surfarray.blit_array(screen, frames[ind].swapaxes(0, 1))
        pygame_widgets.update(events)

        step_text = font.render(f"Step {ind + 1}/{len(frames)}", True, (255, 255, 255))
        bg_rect = step_text.get_rect(topleft=(10, 20))
        pygame.draw.rect(screen, (0, 0, 0), bg_rect.inflate(10, 4))
        screen.blit(step_text, (10, 20))

        pygame.display.update()
