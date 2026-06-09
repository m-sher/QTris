"""Shared max-stat tracking and text-panel helpers for Tetris demos.

Drawing helpers here operate on pygame surfaces/fonts passed in by callers
(unlike rendering.py, which stays numpy-only).
"""

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
    header = small_font.render("Max (episode / run)", True, (255, 255, 255))
    screen.blit(header, (x, y))
    for i, (key, label) in enumerate(_STAT_LABELS):
        row = font.render(
            f"{label}: {episode[key]} / {run[key]}", True, (255, 255, 255)
        )
        screen.blit(row, (x, y + 18 + i * 20))
