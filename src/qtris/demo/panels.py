"""Shared pygame drawing, stat tracking, and replay UI for the Tetris demos.

Everything pygame-coupled that the single-player demos share lives here;
numpy-only image helpers stay in rendering.py.
"""

import pygame
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider

from qtris.demo.constants import BCG_LABELS

WHITE = (255, 255, 255)

# BCG attention panel layout (vertical column along the right edge)
BCG_PANEL_X = 720
BCG_PANEL_Y = 8
BCG_LABEL_H = 20
BCG_HEATMAP_W = 90
BCG_HEATMAP_H = 168
BCG_GAP = 10

_STAT_LABELS = (("b2b", "B2B"), ("combo", "Combo"), ("spike", "Spike"))


class MaxStatTracker:
    """Tracks episode and run maxima for b2b, combo, and spike.

    A spike is the total attack over consecutive attacking placements; it
    resets whenever a placement deals no attack.
    """

    def __init__(self):
        self.episode = {key: 0 for key, _ in _STAT_LABELS}
        self.run = {key: 0 for key, _ in _STAT_LABELS}
        self._spike = 0

    def reset_episode(self):
        self.episode = {key: 0 for key, _ in _STAT_LABELS}
        self._spike = 0

    def update(self, b2b, combo, attack):
        """Fold in one step's values; returns (episode, run) max snapshots."""
        self._spike = self._spike + int(attack) if attack > 0 else 0
        for key, val in (
            ("b2b", int(b2b)),
            ("combo", int(combo)),
            ("spike", self._spike),
        ):
            self.episode[key] = max(self.episode[key], val)
            self.run[key] = max(self.run[key], val)
        return dict(self.episode), dict(self.run)


def draw_max_stats(screen, font, small_font, x, y, episode, run):
    """Draw the max-stats column: header plus B2B/Combo/Spike rows."""
    header = small_font.render("Max (episode / run)", True, WHITE)
    screen.blit(header, (x, y))
    for i, (key, label) in enumerate(_STAT_LABELS):
        row = font.render(f"{label}: {episode[key]} / {run[key]}", True, WHITE)
        screen.blit(row, (x, y + 18 + i * 20))


def draw_text_column(screen, font, x, y, lines, spacing=20):
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, WHITE), (x, y + i * spacing))


def draw_info_panel(
    screen, font, small_font, screen_w, left_lines, state_lines, max_stat_pair, action
):
    """Bottom info panel: rewards | current state | max stats columns + action row."""
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 610, screen_w, 190))
    pygame.draw.line(screen, WHITE, (335, 610), (335, 765), 2)
    pygame.draw.line(screen, WHITE, (590, 610), (590, 765), 2)

    base_y = 615
    draw_text_column(screen, font, 10, base_y, left_lines)
    draw_text_column(screen, font, 345, base_y, state_lines)
    draw_max_stats(screen, font, small_font, 600, base_y, *max_stat_pair)
    screen.blit(font.render(f"Action: {action}", True, WHITE), (10, base_y + 155))


def draw_board_area(screen, board, vis_board, sidebar, scores, garbage, piece_colors):
    """Blit the garbage bar, bordered board, piece sidebar, and attention scores."""
    board_surf = pygame.Surface((10, 24))
    piece_surf = pygame.Surface((5, 28))
    scores_surf = pygame.Surface((5, 12))
    garbage_surf = pygame.Surface((garbage.shape[1], garbage.shape[0]))

    if vis_board is not None:
        colored_board = piece_colors[vis_board[0, ..., 0].numpy()]
        pygame.surfarray.blit_array(board_surf, colored_board.transpose(1, 0, 2))
    else:
        pygame.surfarray.blit_array(board_surf, board[0, ..., 0].numpy().T * 255)

    pygame.surfarray.blit_array(piece_surf, sidebar.transpose(1, 0, 2))
    pygame.surfarray.blit_array(scores_surf, scores.transpose(1, 0, 2))
    pygame.surfarray.blit_array(garbage_surf, garbage.transpose(1, 0, 2))

    board_surf = pygame.transform.scale(board_surf, (250, 600))
    piece_surf = pygame.transform.scale(piece_surf, (125, 600))
    scores_surf = pygame.transform.scale(scores_surf, (250, 600))
    garbage_surf = pygame.transform.scale(garbage_surf, (25, 600))

    board_with_border = pygame.Surface((254, 604))
    board_with_border.fill(WHITE)
    board_with_border.blit(board_surf, (2, 2))

    screen.blit(garbage_surf, (0, 0))
    screen.blit(board_with_border, (25, 0))
    screen.blit(piece_surf, (285, 0))
    screen.blit(scores_surf, (415, 0))


def draw_bcg_panel(screen, small_font, heatmaps, values):
    """Draw the right-edge BCG attention column: label + heatmap per stat."""
    for i in range(3):
        hy = BCG_PANEL_Y + i * (BCG_LABEL_H + BCG_HEATMAP_H + BCG_GAP)
        label = small_font.render(f"{BCG_LABELS[i]}: {values[i]}", True, WHITE)
        screen.blit(label, (BCG_PANEL_X, hy))
        surf = pygame.Surface((5, 12))
        pygame.surfarray.blit_array(surf, heatmaps[i].transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (BCG_HEATMAP_W, BCG_HEATMAP_H))
        screen.blit(surf, (BCG_PANEL_X, hy + BCG_LABEL_H))


def draw_step_counter(screen, font, step, num_steps):
    """Step counter at the top left, on a black backing rect."""
    text = font.render(f"Step: {step}/{num_steps}", True, WHITE)
    rect = text.get_rect()
    rect.topleft = (10, 25)
    pygame.draw.rect(screen, (0, 0, 0), rect.inflate(10, 4))
    screen.blit(text, (10, 25))


def run_replay(screen, font, frames, num_steps, draw_overlay):
    """Replay UI shared by the single-player demos; runs until window close.

    draw_overlay(ind) redraws the per-step overlay (the bottom info panel)
    on top of the recorded frame.
    """
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

    paused = True

    def toggle_pause():
        nonlocal paused
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

        ind = slider.getValue()
        pygame.surfarray.blit_array(screen, frames[ind].swapaxes(0, 1))

        draw_step_counter(screen, font, ind + 1, num_steps)

        speed_text = font.render(f"Speed: {speed_slider.getValue()} FPS", True, WHITE)
        speed_rect = speed_text.get_rect(topleft=(220, 55))
        pygame.draw.rect(screen, (0, 0, 0), speed_rect.inflate(10, 4))
        screen.blit(speed_text, (220, 55))

        pygame_widgets.update(events)

        draw_overlay(ind)

        pygame.display.update()
