"""Scrubbable pygame viewer for the TETR.IO bot replays in tetrio/replays/.

Each replay is `{"frames": [...]}` where a frame holds the 24x10 occupancy board
(before the move), the piece stream `[active, hold, *queue]`, the b2b/combo/garbage
stats, the incoming garbage snapshot, and the move's converted keys. Attack is
reconstructed by replaying each lock through the env scorer; the panel shows
this-piece attack and running APP (attack per used piece). Boards are
occupancy-only, so the stack renders monochrome.

Run: `uv run python scripts/view_replay.py [PATH] [--all] [--fps N]`
PATH defaults to the most recent replay; --all also shows discarded (unused) frames.
"""

import argparse
import glob
import json
import os

import pygame
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider

import numpy as np

from qtris.demo.constants import PIECE_COLORS
from TetrisEnv.Moves import Keys
from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import PyTetrisEnv

REPLAY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tetrio", "replays"
)

PIECE_LETTERS = ["", "I", "J", "L", "O", "S", "T", "Z", "G"]

CELL = 24
COLS, ROWS = 10, 24
BOARD_W, BOARD_H = COLS * CELL, ROWS * CELL
HEADER_H = 64
MARGIN = 12
GAP = 16
PANEL_W = 250
SCREEN_W = MARGIN + BOARD_W + GAP + PANEL_W + MARGIN
SCREEN_H = HEADER_H + BOARD_H + MARGIN
BOARD_X, BOARD_Y = MARGIN, HEADER_H
PANEL_X = BOARD_X + BOARD_W + GAP

WHITE = (255, 255, 255)
EMPTY = (18, 18, 22)
GRID = (40, 40, 48)
STACK = (120, 140, 170)


def latest_replay():
    files = sorted(glob.glob(os.path.join(REPLAY_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No replays found in {REPLAY_DIR}")
    return files[-1]


def load_frames(path, show_all=False):
    """Load a replay's frames, dropping discarded (used==False) frames unless show_all."""
    with open(path) as f:
        frames = json.load(f)["frames"]
    if not show_all:
        kept = [fr for fr in frames if fr.get("used") is not False]
        if kept:
            frames = kept
    if not frames:
        raise SystemExit(f"{path} has no frames to show")
    return frames


def _replay_env():
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=999,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        seed=None,
        idx=0,
        garbage_chance=0,
        auto_push_garbage=False,
        auto_fill_queue=False,
        use_shaping=False,
        placement_candidates=False,
    )
    env.reset()
    return env


def reconstruct_attack(env, frame):
    """Outgoing attack for one pre-lock frame, via the env scorer.

    Replays do not store attack; board + keys + b2b/combo are enough to replay the lock.
    """
    keys = frame.get("key_sequence") or []
    if Keys.HARD_DROP not in keys:
        return 0.0
    pieces = frame["pieces"]
    if len(pieces) < 3:
        return 0.0

    stored = np.asarray(frame["board"], dtype=np.float32)
    board = np.zeros((40, 10), dtype=np.float32)
    rows = min(stored.shape[0], 40)
    board[-rows:] = stored[-rows:]

    env._board = board
    env._vis_board = board.copy()
    env._active_piece = env._spawn_piece(PieceType(int(pieces[0])))
    env._hold_piece = PieceType(int(pieces[1]))
    env._queue = [PieceType(int(p)) for p in pieces[2:]]
    b2b, combo, _ = frame["b2b_combo_garbage"]
    env._scorer._b2b = int(b2b)
    env._scorer._combo = int(combo)

    _, _, attack, _, _, _, _, _, _ = env._execute_action(
        env._board,
        env._vis_board,
        env._active_piece,
        env._hold_piece,
        env._queue,
        np.asarray(keys, dtype=np.int64),
    )
    return float(attack)


def apply_running_app(frames, attacks):
    """Write per-frame attack plus running APP. Only used=True placements count."""
    total = 0.0
    n = 0
    for frame, attack in zip(frames, attacks):
        attack = float(attack)
        frame["attack"] = attack
        if frame.get("used", True) is True:
            total += attack
            n += 1
        frame["total_attack"] = total
        frame["app_pieces"] = n
        frame["app"] = (total / n) if n else 0.0


def annotate_attacks(frames, env=None):
    """Attach reconstructed attack / running APP onto each frame."""
    if env is None:
        env = _replay_env()
    apply_running_app(frames, [reconstruct_attack(env, frame) for frame in frames])
    return frames


def draw_piece(screen, font, x, y, size, val):
    """A colored piece square with its letter; val 0 (none) draws an empty slot."""
    pygame.draw.rect(screen, GRID, (x, y, size, size), border_radius=3)
    if val:
        pygame.draw.rect(
            screen,
            PIECE_COLORS[val],
            (x + 2, y + 2, size - 4, size - 4),
            border_radius=3,
        )
        letter = font.render(PIECE_LETTERS[val], True, (0, 0, 0))
        screen.blit(letter, letter.get_rect(center=(x + size // 2, y + size // 2)))


def draw_board(screen, board):
    pygame.draw.rect(screen, EMPTY, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H))
    for r, row in enumerate(board):
        for c, val in enumerate(row):
            if val:
                pygame.draw.rect(
                    screen,
                    STACK,
                    (BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL - 1, CELL - 1),
                )
    pygame.draw.rect(screen, GRID, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H), 1)


def wrap(text, font, width):
    lines = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if font.size(trial)[0] > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines or [""]


def draw_panel(screen, font, small, frame, ind, total):
    x, y = PANEL_X, BOARD_Y
    pieces = frame["pieces"]
    active, hold = pieces[0], pieces[1]
    nxt = pieces[2:8]
    b2b, combo, total_garbage = frame["b2b_combo_garbage"]
    incoming = len(frame.get("garbage_queue", []))

    screen.blit(font.render(f"Frame {ind + 1}/{total}", True, WHITE), (x, y))
    y += 34

    screen.blit(small.render("Active", True, WHITE), (x, y))
    draw_piece(screen, small, x + 70, y - 4, 28, active)
    y += 34
    screen.blit(small.render("Hold", True, WHITE), (x, y))
    draw_piece(screen, small, x + 70, y - 4, 28, hold)
    y += 38

    screen.blit(small.render("Next", True, WHITE), (x, y))
    y += 22
    for i, val in enumerate(nxt):
        draw_piece(screen, small, x + i * 34, y, 28, val)
    y += 46

    attack = frame.get("attack", 0.0)
    total_attack = frame.get("total_attack", 0.0)
    app = frame.get("app", 0.0)
    app_pieces = frame.get("app_pieces", 0)
    for label in (
        f"B2B: {int(b2b)}",
        f"Combo: {int(combo)}",
        f"Garbage in queue: {incoming}",
        f"Total garbage: {int(total_garbage)}",
        f"Attack: {int(attack)}",
        f"APP: {app:.3f}  ({int(total_attack)}/{app_pieces})",
    ):
        screen.blit(small.render(label, True, WHITE), (x, y))
        y += 24
    y += 8

    screen.blit(small.render("Keys:", True, WHITE), (x, y))
    y += 22
    keys = ", ".join(frame.get("converted_keys") or ["(none)"])
    for line in wrap(keys, small, PANEL_W - 8):
        screen.blit(small.render(line, True, (180, 200, 220)), (x, y))
        y += 20


def main(cli_args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="replay JSON (default: most recent)")
    ap.add_argument("--all", action="store_true", help="include discarded frames")
    ap.add_argument("--fps", type=int, default=8, help="initial playback speed")
    args = ap.parse_args() if cli_args is None else cli_args

    path = args.path or latest_replay()
    frames = annotate_attacks(load_frames(path, show_all=args.all))
    total = len(frames)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Replay: {os.path.basename(path)}")
    font = pygame.font.Font(None, 30)
    small = pygame.font.Font(None, 24)

    slider = Slider(
        screen,
        10,
        12,
        SCREEN_W - 20,
        10,
        min=0,
        max=total - 1,
        step=1,
        colour=(125, 125, 125),
        handleColour=(50, 50, 50),
    )

    paused = True

    def toggle_pause():
        nonlocal paused
        paused = not paused
        play_btn.setText("Play" if paused else "Pause")

    play_btn = Button(
        screen,
        10,
        38,
        60,
        20,
        text="Play",
        fontSize=16,
        margin=0,
        onClick=toggle_pause,
    )

    _back = Button(
        screen,
        76,
        38,
        28,
        20,
        text="<",
        fontSize=16,
        margin=0,
        onClick=lambda: slider.setValue(max(0, slider.getValue() - 1)),
    )

    _fwd = Button(
        screen,
        108,
        38,
        28,
        20,
        text=">",
        fontSize=16,
        margin=0,
        onClick=lambda: slider.setValue(min(total - 1, slider.getValue() + 1)),
    )

    speed = Slider(
        screen,
        150,
        40,
        140,
        10,
        min=1,
        max=60,
        step=1,
        initial=args.fps,
        colour=(125, 125, 125),
        handleColour=(50, 50, 50),
    )

    clock = pygame.time.Clock()
    last_step = pygame.time.get_ticks()
    prev_ind, prev_paused = -1, paused

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if not paused:
            now = pygame.time.get_ticks()
            if now - last_step >= 1000 / speed.getValue():
                cur = slider.getValue()
                if cur < total - 1:
                    slider.setValue(cur + 1)
                    last_step = now
                else:
                    paused = True
                    play_btn.setText("Play")

        ind = slider.getValue()

        if events or not paused or ind != prev_ind or paused != prev_paused:
            screen.fill((0, 0, 0))
            draw_board(screen, frames[ind]["board"])
            draw_panel(screen, font, small, frames[ind], ind, total)
            screen.blit(
                small.render(f"Speed: {speed.getValue()} FPS", True, WHITE), (300, 40)
            )

            pygame_widgets.update(events)
            pygame.display.update()
            prev_ind, prev_paused = ind, paused
        clock.tick(60)


if __name__ == "__main__":
    main()
