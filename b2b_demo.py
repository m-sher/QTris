#!/usr/bin/env python3
"""
B2B Search GUI Demo

Interactive Tetris viewer that uses the b2b_search C extension
to play automatically. Features:
  - Configurable search_depth, beam_width, speed, garbage settings
  - Play / Pause / Step Forward / Step Back / Reset controls
  - Board, hold piece, queue, and live stats display
"""

import pygame
import numpy as np
import copy
import sys
import argparse

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Pieces import PieceType
from TetrisEnv.RotationSystem import RotationSystem

# ── Tetris piece colors (guideline) ─────────────────────────
PIECE_COLORS = {
    0: (30, 30, 30),       # N / empty
    1: (0, 220, 220),      # I  - cyan
    2: (0, 80, 255),       # J  - blue
    3: (255, 150, 0),      # L  - orange
    4: (255, 230, 0),      # O  - yellow
    5: (0, 220, 50),       # S  - green
    6: (180, 0, 255),      # T  - purple
    7: (255, 30, 30),      # Z  - red
    8: (100, 100, 100),    # G  - garbage gray
}

# Piece cell definitions (rot 0) for mini previews — from RotationSystem
PIECE_PREVIEW_CELLS = {
    PieceType.I: [(1, 0), (1, 1), (1, 2), (1, 3)],
    PieceType.J: [(0, 0), (1, 0), (1, 1), (1, 2)],
    PieceType.L: [(0, 2), (1, 0), (1, 1), (1, 2)],
    PieceType.O: [(0, 0), (0, 1), (1, 0), (1, 1)],
    PieceType.S: [(0, 1), (0, 2), (1, 0), (1, 1)],
    PieceType.T: [(0, 1), (1, 0), (1, 1), (1, 2)],
    PieceType.Z: [(0, 0), (0, 1), (1, 1), (1, 2)],
}

# ── Layout constants ─────────────────────────────────────────
CELL = 28
MINI_CELL = 18
BOARD_VISIBLE_ROWS = 20
BOARD_HIDDEN_ROWS = 4
BOARD_ROWS = 24
BOARD_COLS = 10

LEFT_PANEL_W = 260
BOARD_W = BOARD_COLS * CELL
BOARD_H = BOARD_ROWS * CELL
RIGHT_PANEL_W = 200
WIN_W = LEFT_PANEL_W + BOARD_W + RIGHT_PANEL_W
WIN_H = BOARD_H + 40  # small bottom margin

# Colors
BG = (18, 18, 24)
PANEL_BG = (28, 28, 36)
GRID_LINE = (45, 45, 55)
HIDDEN_OVERLAY = (0, 0, 0, 140)
TEXT_COLOR = (220, 220, 230)
ACCENT = (80, 180, 255)
BTN_BG = (50, 50, 65)
BTN_HOVER = (70, 70, 90)
BTN_ACTIVE = (90, 130, 200)
LABEL_DIM = (140, 140, 160)
BORDER_COLOR = (60, 60, 75)


# ── Snapshot helpers ─────────────────────────────────────────
def capture_snapshot(env):
    return {
        "board": env._board.copy(),
        "vis_board": env._vis_board.copy(),
        "active_piece": copy.deepcopy(env._active_piece),
        "hold_piece": copy.deepcopy(env._hold_piece),
        "queue": copy.deepcopy(env._queue),
        "b2b": env._scorer._b2b,
        "combo": env._scorer._combo,
        "garbage_queue": copy.deepcopy(env._garbage_queue),
        "step_num": env._step_num,
        "episode_ended": env._episode_ended,
        "last_phi": env._last_phi,
        "next_bag": copy.deepcopy(env._next_bag),
        "random_state": env._random.getstate(),
        "rng_t": env._tetrio_rng._t,
    }


def restore_snapshot(env, snap):
    env._board = snap["board"].copy()
    env._vis_board = snap["vis_board"].copy()
    env._active_piece = copy.deepcopy(snap["active_piece"])
    env._hold_piece = copy.deepcopy(snap["hold_piece"])
    env._queue = copy.deepcopy(snap["queue"])
    env._scorer._b2b = snap["b2b"]
    env._scorer._combo = snap["combo"]
    env._garbage_queue = copy.deepcopy(snap["garbage_queue"])
    env._step_num = snap["step_num"]
    env._episode_ended = snap["episode_ended"]
    env._last_phi = snap["last_phi"]
    env._next_bag = copy.deepcopy(snap["next_bag"])
    env._random.setstate(snap["random_state"])
    env._tetrio_rng._t = snap["rng_t"]


# ── UI Widgets ───────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, font, toggle=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.toggle = toggle
        self.active = False
        self.hovered = False

    def draw(self, surface):
        if self.active:
            color = BTN_ACTIVE
        elif self.hovered:
            color = BTN_HOVER
        else:
            color = BTN_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, 1, border_radius=5)
        txt = self.font.render(self.label, True, TEXT_COLOR)
        tx = self.rect.centerx - txt.get_width() // 2
        ty = self.rect.centery - txt.get_height() // 2
        surface.blit(txt, (tx, ty))

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class Spinner:
    """Integer value with +/- buttons."""

    def __init__(self, x, y, label, value, lo, hi, step, font, small_font):
        self.label = label
        self.value = value
        self.lo = lo
        self.hi = hi
        self.step = step
        self.font = font
        self.small_font = small_font
        self.minus_rect = pygame.Rect(x, y, 28, 26)
        self.plus_rect = pygame.Rect(x + 130, y, 28, 26)
        self.label_y = y
        self.x = x

    def draw(self, surface):
        # Label above
        lbl = self.small_font.render(self.label, True, LABEL_DIM)
        surface.blit(lbl, (self.x, self.label_y - 16))
        # Minus button
        pygame.draw.rect(surface, BTN_BG, self.minus_rect, border_radius=4)
        pygame.draw.rect(surface, BORDER_COLOR, self.minus_rect, 1, border_radius=4)
        m = self.font.render("-", True, TEXT_COLOR)
        surface.blit(m, (self.minus_rect.centerx - m.get_width() // 2,
                         self.minus_rect.centery - m.get_height() // 2))
        # Value
        val_str = f"{self.value}"
        if isinstance(self.value, float):
            val_str = f"{self.value:.2f}"
        v = self.font.render(val_str, True, TEXT_COLOR)
        surface.blit(v, (self.x + 80 - v.get_width() // 2,
                         self.minus_rect.centery - v.get_height() // 2))
        # Plus button
        pygame.draw.rect(surface, BTN_BG, self.plus_rect, border_radius=4)
        pygame.draw.rect(surface, BORDER_COLOR, self.plus_rect, 1, border_radius=4)
        p = self.font.render("+", True, TEXT_COLOR)
        surface.blit(p, (self.plus_rect.centerx - p.get_width() // 2,
                         self.plus_rect.centery - p.get_height() // 2))

    def handle_click(self, pos):
        if self.minus_rect.collidepoint(pos):
            if isinstance(self.value, float):
                self.value = round(max(self.lo, self.value - self.step), 2)
            else:
                self.value = max(self.lo, self.value - self.step)
            return True
        if self.plus_rect.collidepoint(pos):
            if isinstance(self.value, float):
                self.value = round(min(self.hi, self.value + self.step), 2)
            else:
                self.value = min(self.hi, self.value + self.step)
            return True
        return False


# ── Drawing helpers ──────────────────────────────────────────
def draw_board(surface, vis_board, ox, oy):
    """Draw the 24-row board with grid lines."""
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            x = ox + c * CELL
            y = oy + r * CELL
            val = int(vis_board[r, c])
            color = PIECE_COLORS.get(val, PIECE_COLORS[0])
            if val == 0:
                color = (12, 12, 18) if r >= BOARD_HIDDEN_ROWS else (6, 6, 10)
            pygame.draw.rect(surface, color, (x + 1, y + 1, CELL - 2, CELL - 2),
                             border_radius=3)
    # Grid lines
    for c in range(BOARD_COLS + 1):
        x = ox + c * CELL
        pygame.draw.line(surface, GRID_LINE, (x, oy), (x, oy + BOARD_H))
    for r in range(BOARD_ROWS + 1):
        y = oy + r * CELL
        pygame.draw.line(surface, GRID_LINE, (ox, y), (ox + BOARD_W, y))
    # Hidden row overlay
    overlay = pygame.Surface((BOARD_W, BOARD_HIDDEN_ROWS * CELL), pygame.SRCALPHA)
    overlay.fill(HIDDEN_OVERLAY)
    surface.blit(overlay, (ox, oy))
    # Border
    pygame.draw.rect(surface, BORDER_COLOR, (ox - 1, oy - 1, BOARD_W + 2, BOARD_H + 2), 2)


def draw_mini_piece(surface, piece_type, x, y, cell_size=MINI_CELL):
    """Draw a small piece preview."""
    if piece_type == PieceType.N or piece_type not in PIECE_PREVIEW_CELLS:
        # Draw empty box
        pygame.draw.rect(surface, (40, 40, 50),
                         (x, y, cell_size * 4, cell_size * 2), 1, border_radius=3)
        return
    cells = PIECE_PREVIEW_CELLS[piece_type]
    color = PIECE_COLORS[piece_type.value]
    # Center the piece horizontally
    cols = [c for _, c in cells]
    min_c, max_c = min(cols), max(cols)
    w = max_c - min_c + 1
    offset_x = (4 - w) * cell_size // 2
    rows = [r for r, _ in cells]
    min_r = min(rows)
    for r, c in cells:
        px = x + offset_x + (c - min_c) * cell_size
        py = y + (r - min_r) * cell_size
        pygame.draw.rect(surface, color,
                         (px + 1, py + 1, cell_size - 2, cell_size - 2),
                         border_radius=2)


def draw_active_ghost(surface, env, ox, oy):
    """Draw the active piece's ghost (hard-drop preview) on the board."""
    piece = env._active_piece
    rs = RotationSystem()
    cells = rs.orientations[piece.piece_type][piece.r]
    # Find hard-drop position
    loc = piece.loc.copy()
    from TetrisEnv.helpers import overlaps
    while not overlaps(cells=cells, loc=loc + [1, 0], board=env._board):
        loc = loc + [1, 0]
    color = PIECE_COLORS.get(piece.piece_type.value, (200, 200, 200))
    ghost_color = (color[0] // 3, color[1] // 3, color[2] // 3)
    for cr, cc in cells:
        r = loc[0] + cr
        c = loc[1] + cc
        if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
            x = ox + c * CELL
            y = oy + r * CELL
            pygame.draw.rect(surface, ghost_color,
                             (x + 2, y + 2, CELL - 4, CELL - 4), 2, border_radius=3)


# ── Headless runner ──────────────────────────────────────────
def run_headless(args):
    """Run the game without a GUI, printing per-turn stats to stdout."""
    env = PyTetrisEnv(
        queue_size=args.queue_size,
        max_holes=None,
        max_height=20,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        seed=args.seed,
        idx=0,
        garbage_chance=args.garbage_chance,
        garbage_min=args.garbage_min,
        garbage_max=args.garbage_max,
        garbage_push_delay=args.garbage_delay,
        auto_push_garbage=True,
        auto_fill_queue=True,
    )
    env.reset()
    searcher = CB2BSearch()

    total_attack = 0.0
    max_b2b = 0
    max_combo = 0
    max_attack = 0.0
    max_consec_attack = 0.0
    cur_consec_attack = 0.0

    hdr = (
        f"{'Turn':>5}  {'B2B':>4}  {'Combo':>5}  {'Attack':>6}  "
        f"{'MaxB2B':>6}  {'MaxCmb':>6}  {'MaxAtk':>6}  "
        f"{'MaxCon':>6}  {'TotAtk':>7}  "
        f"{'MaxHt':>5}  {'AvgHt':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    for step in range(1, args.num_steps + 1):
        board = env._board
        active = env._active_piece.piece_type.value
        hold = env._hold_piece.value
        queue_types = np.array([p.value for p in env._queue], dtype=np.int32)
        b2b = env._scorer._b2b
        combo = env._scorer._combo

        total_garb = env._get_total_garbage()

        action_idx, sequence = searcher.search(
            board=board,
            active_piece=active,
            hold_piece=hold,
            queue=queue_types,
            b2b=b2b,
            combo=combo,
            total_garbage=total_garb,
            garbage_push_delay=env._garbage_push_delay,
            search_depth=args.search_depth,
            beam_width=args.beam_width,
            max_len=15,
        )

        if action_idx < 0:
            print(f"\n** No valid move found at turn {step} — game over **")
            break

        ts = env._step(sequence)

        atk = float(ts.reward["attack"])
        total_attack += atk

        # Track consecutive-attack streak
        if atk > 0:
            cur_consec_attack += atk
        else:
            max_consec_attack = max(max_consec_attack, cur_consec_attack)
            cur_consec_attack = 0.0

        cur_b2b = env._scorer._b2b
        cur_combo = env._scorer._combo
        max_b2b = max(max_b2b, cur_b2b)
        max_combo = max(max_combo, cur_combo)
        max_attack = max(max_attack, atk)

        # Compute per-column heights
        col_heights = np.zeros(env._board.shape[1], dtype=int)
        for c in range(env._board.shape[1]):
            for r in range(env._board.shape[0]):
                if env._board[r, c] != 0:
                    col_heights[c] = env._board.shape[0] - r
                    break
        max_height = int(col_heights.max())
        avg_height = float(col_heights.mean())

        print(
            f"{step:>5}  {cur_b2b:>4}  {cur_combo:>5}  {atk:>6.0f}  "
            f"{max_b2b:>6}  {max_combo:>6}  {max_attack:>6.0f}  "
            f"{max(max_consec_attack, cur_consec_attack):>6.0f}  "
            f"{total_attack:>7.0f}  "
            f"{max_height:>5}  {avg_height:>5.1f}"
        )

        if ts.is_last():
            print(f"\n** Game over at turn {step} **")
            break

    # Flush last streak
    max_consec_attack = max(max_consec_attack, cur_consec_attack)

    print()
    print("=" * 40)
    print(f"  Total turns:         {step}")
    print(f"  Total attack:        {total_attack:.0f}")
    print(f"  Max B2B reached:     {max_b2b}")
    print(f"  Max combo reached:   {max_combo}")
    print(f"  Max single attack:   {max_attack:.0f}")
    print(f"  Max consec. attack:  {max_consec_attack:.0f}")
    print(f"  Attack/piece:        {total_attack / max(step, 1):.3f}")
    print("=" * 40)


# ── Main ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="B2B Search GUI Demo")
    ap.add_argument('--headless', action='store_true',
                    help='Run without GUI, print per-turn stats to terminal')
    ap.add_argument('--search-depth', type=int, default=7)
    ap.add_argument('--beam-width', type=int, default=128)
    ap.add_argument('--num-steps', type=int, default=200,
                    help='Max turns to play (headless mode)')
    ap.add_argument('--queue-size', type=int, default=5)
    ap.add_argument('--garbage-chance', type=float, default=0.15)
    ap.add_argument('--garbage-min', type=int, default=1)
    ap.add_argument('--garbage-max', type=int, default=4)
    ap.add_argument('--garbage-delay', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.headless:
        run_headless(args)
        return

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("B2B Search Demo")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 16, bold=True)
    small_font = pygame.font.SysFont("monospace", 13)
    big_font = pygame.font.SysFont("monospace", 22, bold=True)
    stat_font = pygame.font.SysFont("monospace", 15)

    # ── Config spinners ──────────────────────────────────────
    sp_x = 18
    sp_start_y = 130
    sp_gap = 50

    spinners = {
        "search_depth": Spinner(sp_x, sp_start_y, "Search Depth",
                                4, 1, 16, 1, font, small_font),
        "beam_width": Spinner(sp_x, sp_start_y + sp_gap, "Beam Width",
                              64, 4, 256, 4, font, small_font),
        "speed": Spinner(sp_x, sp_start_y + sp_gap * 2, "Speed (steps/s)",
                         5, 1, 60, 1, font, small_font),
        "queue_size": Spinner(sp_x, sp_start_y + sp_gap * 3, "Queue Size",
                              5, 3, 12, 1, font, small_font),
        "garbage_chance": Spinner(sp_x, sp_start_y + sp_gap * 4, "Garbage Chance",
                                  0.15, 0.0, 1.0, 0.05, font, small_font),
        "garbage_min": Spinner(sp_x, sp_start_y + sp_gap * 5, "Garbage Min",
                               1, 0, 10, 1, font, small_font),
        "garbage_max": Spinner(sp_x, sp_start_y + sp_gap * 6, "Garbage Max",
                               4, 0, 10, 1, font, small_font),
        "garbage_delay": Spinner(sp_x, sp_start_y + sp_gap * 7, "Garbage Delay",
                                  1, 0, 5, 1, font, small_font),
        "seed": Spinner(sp_x, sp_start_y + sp_gap * 8, "Seed",
                        42, 0, 9999, 1, font, small_font),
    }

    # ── Buttons ──────────────────────────────────────────────
    btn_y = 18
    btn_h = 32
    btn_gap = 6

    btn_play = Button((sp_x, btn_y, 100, btn_h), "Play", font, toggle=True)
    btn_step_fwd = Button((sp_x + 106, btn_y, 70, btn_h), "Step>", font)
    btn_step_back = Button((sp_x + 106 + 76, btn_y, 70, btn_h), "<Step", font)
    btn_reset = Button((sp_x, btn_y + btn_h + btn_gap, 100, btn_h), "Reset", font)
    btn_restart = Button((sp_x + 106, btn_y + btn_h + btn_gap, 140, btn_h),
                         "New Game", font)

    buttons = [btn_play, btn_step_fwd, btn_step_back, btn_reset, btn_restart]

    # ── State ────────────────────────────────────────────────
    searcher = CB2BSearch()

    def create_env():
        qs = spinners["queue_size"].value
        env = PyTetrisEnv(
            queue_size=qs,
            max_holes=None,
            max_height=20,
            max_steps=None,
            max_len=15,
            pathfinding=False,
            seed=spinners["seed"].value,
            idx=0,
            garbage_chance=spinners["garbage_chance"].value,
            garbage_min=spinners["garbage_min"].value,
            garbage_max=spinners["garbage_max"].value,
            garbage_push_delay=spinners["garbage_delay"].value,
            auto_push_garbage=True,
            auto_fill_queue=True,
        )
        env.reset()
        return env

    env = create_env()
    history = [capture_snapshot(env)]
    history_idx = 0
    last_action_idx = -1
    last_sequence = None
    last_attack = 0.0
    last_clears = 0
    total_attack = 0.0
    total_clears = 0
    playing = False
    step_timer = 0.0
    game_over = False
    step_info_history = [{"attack": 0, "clears": 0, "action_idx": -1}]

    def do_step_forward():
        nonlocal history, history_idx, last_action_idx, last_sequence
        nonlocal last_attack, last_clears, total_attack, total_clears, game_over

        if game_over:
            return

        # If we stepped back, truncate future history
        if history_idx < len(history) - 1:
            history = history[:history_idx + 1]
            step_info_history_truncate = len(history)

        # Run b2b search
        board = env._board
        active = env._active_piece.piece_type.value
        hold = env._hold_piece.value
        queue_types = np.array([p.value for p in env._queue], dtype=np.int32)
        b2b = env._scorer._b2b
        combo = env._scorer._combo
        total_garb = env._get_total_garbage()

        sd = spinners["search_depth"].value
        bw = spinners["beam_width"].value

        action_idx, sequence = searcher.search(
            board=board,
            active_piece=active,
            hold_piece=hold,
            queue=queue_types,
            b2b=b2b,
            combo=combo,
            total_garbage=total_garb,
            garbage_push_delay=env._garbage_push_delay,
            search_depth=sd,
            beam_width=bw,
            max_len=15,
        )

        last_action_idx = action_idx
        last_sequence = sequence

        # Execute action
        ts = env._step(sequence)

        # Extract reward info
        atk = float(ts.reward["attack"])
        clr = float(ts.reward["clear"])
        last_attack = atk
        last_clears = int(clr)
        total_attack += atk
        total_clears += int(clr)

        info = {"attack": atk, "clears": int(clr), "action_idx": action_idx}
        step_info_history.append(info)

        # Check game over
        if ts.is_last():
            game_over = True

        # Save snapshot
        history.append(capture_snapshot(env))
        history_idx = len(history) - 1

    def do_step_back():
        nonlocal history_idx, last_attack, last_clears, game_over
        if history_idx > 0:
            history_idx -= 1
            restore_snapshot(env, history[history_idx])
            game_over = False
            info = step_info_history[history_idx]
            last_attack = info["attack"]
            last_clears = info["clears"]

    def do_reset():
        nonlocal history, history_idx, game_over, playing
        nonlocal last_attack, last_clears, total_attack, total_clears
        nonlocal step_info_history, last_action_idx, last_sequence
        if history:
            restore_snapshot(env, history[0])
        history = [capture_snapshot(env)]
        history_idx = 0
        game_over = False
        playing = False
        btn_play.active = False
        btn_play.label = "Play"
        last_attack = 0
        last_clears = 0
        total_attack = 0
        total_clears = 0
        last_action_idx = -1
        last_sequence = None
        step_info_history = [{"attack": 0, "clears": 0, "action_idx": -1}]

    def do_new_game():
        nonlocal env, history, history_idx, game_over, playing
        nonlocal last_attack, last_clears, total_attack, total_clears
        nonlocal step_info_history, last_action_idx, last_sequence
        env = create_env()
        history = [capture_snapshot(env)]
        history_idx = 0
        game_over = False
        playing = False
        btn_play.active = False
        btn_play.label = "Play"
        last_attack = 0
        last_clears = 0
        total_attack = 0
        total_clears = 0
        last_action_idx = -1
        last_sequence = None
        step_info_history = [{"attack": 0, "clears": 0, "action_idx": -1}]

    # ── Main loop ────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for btn in buttons:
            btn.check_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                    btn_play.active = playing
                    btn_play.label = "Pause" if playing else "Play"
                elif event.key == pygame.K_RIGHT:
                    do_step_forward()
                elif event.key == pygame.K_LEFT:
                    do_step_back()
                elif event.key == pygame.K_r:
                    do_reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_play.clicked(mouse_pos):
                    playing = not playing
                    btn_play.active = playing
                    btn_play.label = "Pause" if playing else "Play"
                elif btn_step_fwd.clicked(mouse_pos):
                    do_step_forward()
                elif btn_step_back.clicked(mouse_pos):
                    do_step_back()
                elif btn_reset.clicked(mouse_pos):
                    do_reset()
                elif btn_restart.clicked(mouse_pos):
                    do_new_game()
                else:
                    for sp in spinners.values():
                        sp.handle_click(mouse_pos)

        # Auto-play — at most one step per frame so the screen always refreshes
        if playing and not game_over:
            speed = spinners["speed"].value
            step_timer += dt
            interval = 1.0 / speed if speed > 0 else 1.0
            if step_timer >= interval:
                step_timer = min(step_timer - interval, interval)  # don't accumulate
                do_step_forward()
        else:
            step_timer = 0.0

        if game_over and playing:
            playing = False
            btn_play.active = False
            btn_play.label = "Play"

        # ── Draw ─────────────────────────────────────────────
        screen.fill(BG)

        # Left panel background
        pygame.draw.rect(screen, PANEL_BG, (0, 0, LEFT_PANEL_W, WIN_H))
        pygame.draw.line(screen, BORDER_COLOR, (LEFT_PANEL_W, 0),
                         (LEFT_PANEL_W, WIN_H))

        # Right panel background
        rp_x = LEFT_PANEL_W + BOARD_W
        pygame.draw.rect(screen, PANEL_BG, (rp_x, 0, RIGHT_PANEL_W, WIN_H))
        pygame.draw.line(screen, BORDER_COLOR, (rp_x, 0), (rp_x, WIN_H))

        # Buttons
        for btn in buttons:
            btn.draw(screen)

        # Spinners
        for sp in spinners.values():
            sp.draw(screen)

        # ── Board ────────────────────────────────────────────
        board_ox = LEFT_PANEL_W
        board_oy = 5
        draw_board(screen, env._vis_board, board_ox, board_oy)

        # Ghost piece
        if not game_over:
            draw_active_ghost(screen, env, board_ox, board_oy)

        # ── Right panel: Hold + Queue + Stats ────────────────
        rx = rp_x + 16
        ry = 16

        # Hold
        hold_lbl = big_font.render("HOLD", True, ACCENT)
        screen.blit(hold_lbl, (rx, ry))
        ry += 28
        draw_mini_piece(screen, env._hold_piece, rx, ry, MINI_CELL)
        ry += MINI_CELL * 2 + 16

        # Queue
        queue_lbl = big_font.render("NEXT", True, ACCENT)
        screen.blit(queue_lbl, (rx, ry))
        ry += 28
        for i, pt in enumerate(env._queue):
            draw_mini_piece(screen, pt, rx, ry, MINI_CELL)
            ry += MINI_CELL * 2 + 8
        ry += 8

        # Separator
        pygame.draw.line(screen, BORDER_COLOR, (rx, ry), (rx + RIGHT_PANEL_W - 32, ry))
        ry += 12

        # Stats
        stats_lbl = big_font.render("STATS", True, ACCENT)
        screen.blit(stats_lbl, (rx, ry))
        ry += 28

        stat_lines = [
            ("Step", f"{env._step_num}"),
            ("B2B", f"{env._scorer._b2b}"),
            ("Combo", f"{env._scorer._combo}"),
            ("Last Atk", f"{last_attack:.0f}"),
            ("Last Clr", f"{last_clears}"),
            ("Total Atk", f"{total_attack:.0f}"),
            ("Total Clr", f"{total_clears}"),
            ("Garbage Q", f"{env._get_total_garbage()}"),
            ("History", f"{history_idx}/{len(history)-1}"),
        ]

        for label, value in stat_lines:
            l = stat_font.render(f"{label}:", True, LABEL_DIM)
            v = stat_font.render(value, True, TEXT_COLOR)
            screen.blit(l, (rx, ry))
            screen.blit(v, (rx + 100, ry))
            ry += 20

        # Active piece info
        ry += 8
        piece_name = env._active_piece.piece_type.name
        act_lbl = stat_font.render(f"Active: {piece_name}", True, TEXT_COLOR)
        screen.blit(act_lbl, (rx, ry))
        ry += 20

        if last_action_idx >= 0:
            is_hold = last_action_idx // 80
            rem = last_action_idx % 80
            rot = rem // 20
            rem2 = rem % 20
            nc = rem2 // 2
            is_spin = rem2 % 2
            act_str = f"H={is_hold} R={rot} C={nc} Sp={is_spin}"
            act_detail = stat_font.render(act_str, True, LABEL_DIM)
            screen.blit(act_detail, (rx, ry))

        # Game over overlay
        if game_over:
            overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (board_ox, board_oy))
            go_text = big_font.render("GAME OVER", True, (255, 60, 60))
            gx = board_ox + BOARD_W // 2 - go_text.get_width() // 2
            gy = board_oy + BOARD_H // 2 - go_text.get_height() // 2
            screen.blit(go_text, (gx, gy))

        # Bottom bar: keyboard shortcuts
        help_text = small_font.render(
            "Space=Play/Pause  Left/Right=Step  R=Reset  Esc=Quit", True, LABEL_DIM
        )
        screen.blit(help_text, (LEFT_PANEL_W + 10, WIN_H - 20))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
