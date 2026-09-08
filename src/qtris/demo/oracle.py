"""Watch a beam oracle play on its own, with no policy network in the loop."""

import time
import warnings

import numpy as np
from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetrisEnv import PyTetrisEnv

from qtris.demo.constants import PIECE_COLORS, PIECE_DISPLAY
from qtris.training.placement_az import _load_trace_pools

BOARD_ROWS, BOARD_COLS = 24, 10
CELL = 25
BOARD_W, BOARD_H = BOARD_COLS * CELL, BOARD_ROWS * CELL
GARBAGE_W, SIDEBAR_W, DIST_W = 25, 125, 250
SCREEN_W = GARBAGE_W + BOARD_W + SIDEBAR_W + DIST_W + 40
SCREEN_H = BOARD_H + 200
WHITE = (255, 255, 255)


def _quiet_low_occupancy():
    """Drop numba's per-launch occupancy warning."""
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def build_oracle(kind, env, search_depth, beam_width, queue_size, max_len):
    """The named beam engine, exposing CB2BSearch.search_with_scores."""
    if kind == "gpu":
        from qtris.search.gpu_oracle import GpuOracle

        return GpuOracle(
            env,
            search_depth=search_depth,
            beam_width=beam_width,
            queue_size=queue_size,
            max_len=max_len,
        )
    from TetrisEnv.CB2BSearch import CB2BSearch

    return CB2BSearch()


class Stats:
    """Running attack, clear and back-to-back totals across a run of episodes."""

    def __init__(self):
        self.episodes = 0
        self.reset_episode()
        self.run_attack = 0
        self.run_pieces = 0
        self.max_b2b_run = -1

    def reset_episode(self):
        self.pieces = 0
        self.attack = 0
        self.clears = 0
        self.max_b2b = -1
        self.max_combo = -1

    def record(self, attack, clears, b2b, combo):
        self.pieces += 1
        self.run_pieces += 1
        self.attack += attack
        self.run_attack += attack
        self.clears += clears
        self.max_b2b = max(self.max_b2b, b2b)
        self.max_combo = max(self.max_combo, combo)
        self.max_b2b_run = max(self.max_b2b_run, b2b)

    @property
    def app(self):
        return self.attack / self.pieces if self.pieces else 0.0

    @property
    def run_app(self):
        return self.run_attack / self.run_pieces if self.run_pieces else 0.0


def resolve_traces(args):
    """(trace pool, tier name) named by --garbage-traces, or (None, None)."""
    traces_dir = getattr(args, "garbage_traces", None)
    if not traces_dir:
        return None, None
    pools = _load_trace_pools(traces_dir)
    tier = getattr(args, "trace_tier", None) or (list(pools)[-1] if pools else None)
    if tier not in pools:
        raise SystemExit(
            f"trace tier {tier!r} not found in {traces_dir} (have {list(pools)})"
        )
    return pools[tier], tier


def make_env(args, seed, traces=None):
    """A single-player env at the demo's own settings."""
    return PyTetrisEnv(
        queue_size=args.queue_size,
        max_holes=50,
        max_steps=None,
        max_len=args.max_len,
        pathfinding=False,
        garbage_chance=args.garbage_chance,
        garbage_min=1,
        garbage_max=4,
        seed=seed,
        idx=0,
        garbage_traces=traces,
    )


def search(oracle, env, args):
    """Oracle output for the env's current position."""
    queue = np.array([p.value for p in env._queue], dtype=np.int32)
    return oracle.search_with_scores(
        board=env._board,
        active_piece=env._active_piece.piece_type.value,
        hold_piece=env._hold_piece.value,
        queue=queue,
        b2b=int(env._scorer._b2b),
        combo=int(env._scorer._combo),
        total_garbage=int(env._get_total_garbage()),
        garbage_push_delay=env._garbage_push_delay,
        search_depth=args.search_depth,
        beam_width=args.beam_width,
        max_len=args.max_len,
    )


def commit(env, sequence, stats):
    """Step the env with one key sequence and fold the result into the statistics."""
    time_step = env._step(sequence)
    attack = int(time_step.reward["attack"])
    clears = int(time_step.reward["clear"])
    stats.record(attack, clears, int(env._scorer._b2b), int(env._scorer._combo))
    return attack, clears


def candidate_distribution(scores, temperature):
    """Softmax over the root scores, for the candidate strip."""
    if len(scores) == 0:
        return np.zeros(0, dtype=np.float32)
    z = np.asarray(scores, np.float64) / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def run_headless(args):
    """Play without a window, printing one row per turn."""
    seed = args.seed
    traces, tier = resolve_traces(args)
    env = make_env(args, seed, traces)
    env.reset()
    oracle = build_oracle(
        args.oracle,
        env,
        args.search_depth,
        args.beam_width,
        args.queue_size,
        args.max_len,
    )
    stats = Stats()
    source = f"traces:{tier}" if traces else f"chance:{args.garbage_chance}"
    print(
        f"oracle={args.oracle} d{args.search_depth} w{args.beam_width} "
        f"q{args.queue_size} garbage={source}"
    )
    print(
        f"{'turn':>5} {'ms':>7} {'cand':>5} {'atk':>4} {'clr':>4} "
        f"{'b2b':>4} {'cmb':>4} {'app':>6} {'maxh':>5}"
    )
    for turn in range(args.num_steps):
        t0 = time.perf_counter()
        action, sequence, cand_actions, _cs, _seqs, _rows, _v = search(
            oracle, env, args
        )
        elapsed = (time.perf_counter() - t0) * 1e3
        if action < 0:
            sequence = np.full(args.max_len, Keys.PAD, dtype=np.int64)
            sequence[0], sequence[1] = Keys.START, Keys.HARD_DROP
        attack, clears = commit(env, sequence, stats)
        heights = BOARD_ROWS - np.argmax(
            np.vstack([env._board[-BOARD_ROWS:], np.ones((1, BOARD_COLS))]) > 0, axis=0
        )
        print(
            f"{turn:5d} {elapsed:7.1f} {len(cand_actions):5d} {attack:4d} {clears:4d} "
            f"{int(env._scorer._b2b):4d} {int(env._scorer._combo):4d} "
            f"{stats.app:6.3f} {int(heights.max()):5d}"
        )
        if env._is_top_out(env._board):
            stats.episodes += 1
            print(
                f"  died at turn {turn}, APP {stats.app:.3f}, max b2b {stats.max_b2b}"
            )
            seed += 9973
            env = make_env(args, seed, traces)
            env.reset()
            oracle = build_oracle(
                args.oracle,
                env,
                args.search_depth,
                args.beam_width,
                args.queue_size,
                args.max_len,
            )
            stats.reset_episode()
    print(
        f"done: {stats.episodes} deaths, run APP {stats.run_app:.3f}, "
        f"run max b2b {stats.max_b2b_run}"
    )


def _blit_board(screen, pygame, env):
    """Board, garbage bar and the death envelope outline."""
    vis = env._vis_board[-BOARD_ROWS:]
    surf = pygame.Surface((BOARD_COLS, BOARD_ROWS))
    pygame.surfarray.blit_array(surf, PIECE_COLORS[vis].transpose(1, 0, 2))
    surf = pygame.transform.scale(surf, (BOARD_W, BOARD_H))
    border = pygame.Surface((BOARD_W + 4, BOARD_H + 4))
    border.fill(WHITE)
    border.blit(surf, (2, 2))

    pending = int(env._get_total_garbage())
    bar = pygame.Surface((1, BOARD_ROWS))
    col = np.zeros((1, BOARD_ROWS, 3), np.uint8)
    if pending:
        col[0, -min(pending, BOARD_ROWS) :] = (255, 60, 60)
    pygame.surfarray.blit_array(bar, col)
    screen.blit(pygame.transform.scale(bar, (GARBAGE_W, BOARD_H)), (0, 0))
    screen.blit(border, (GARBAGE_W, 0))


def _blit_pieces(screen, pygame, font, env):
    """Hold piece above the visible queue, each labelled."""
    x = GARBAGE_W + BOARD_W + 8
    ids = [env._hold_piece.value] + [p.value for p in env._queue]
    screen.blit(font.render("hold", True, WHITE), (x, 10))
    screen.blit(font.render("next", True, WHITE), (x, 96))
    for slot, pid in enumerate(ids):
        glyph = PIECE_DISPLAY[pid]
        surf = pygame.Surface((glyph.shape[1], glyph.shape[0]))
        pygame.surfarray.blit_array(
            surf,
            (PIECE_COLORS[pid] * glyph[..., None]).astype(np.uint8).transpose(1, 0, 2),
        )
        surf = pygame.transform.scale(surf, (SIDEBAR_W - 20, 60))
        screen.blit(surf, (x, 30 + slot * 78 + (48 if slot else 0)))


def _blit_distribution(screen, pygame, font, probs, best_slot):
    """Candidate strip: one bar per root placement, the chosen one highlighted."""
    x = GARBAGE_W + BOARD_W + SIDEBAR_W + 16
    screen.blit(font.render("candidates", True, WHITE), (x, 4))
    if len(probs) == 0:
        return
    top = np.argsort(-probs)[: min(len(probs), 28)]
    scale = float(probs[top[0]]) or 1.0
    for row, idx in enumerate(top):
        width = int(DIST_W * float(probs[idx]) / scale)
        color = (90, 220, 120) if idx == best_slot else (70, 110, 200)
        pygame.draw.rect(
            screen, color, pygame.Rect(x, 26 + row * 20, max(width, 2), 14)
        )


def _blit_panel(screen, pygame, font, lines):
    """Bottom text panel."""
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, BOARD_H + 8, SCREEN_W, 190))
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, WHITE), (12, BOARD_H + 16 + i * 22))


def run_window(args):
    """Play in a pygame window driven entirely by the oracle."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"beam oracle ({args.oracle})")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    seed = args.seed
    traces, _tier = resolve_traces(args)
    env = make_env(args, seed, traces)
    env.reset()
    oracle = build_oracle(
        args.oracle,
        env,
        args.search_depth,
        args.beam_width,
        args.queue_size,
        args.max_len,
    )
    stats = Stats()
    paused = False
    last_ms = 0.0
    probs = np.zeros(0, np.float32)
    best_slot = -1
    attack = clears = 0

    for turn in range(args.num_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused
        if paused:
            clock.tick(30)
            continue

        t0 = time.perf_counter()
        action, sequence, cand_actions, cand_scores, _seqs, _rows, _v = search(
            oracle, env, args
        )
        last_ms = (time.perf_counter() - t0) * 1e3
        probs = candidate_distribution(cand_scores, args.dist_temp)
        hit = np.flatnonzero(np.asarray(cand_actions, np.int32) == int(action))
        best_slot = int(hit[0]) if hit.size else -1
        if action < 0:
            sequence = np.full(args.max_len, Keys.PAD, dtype=np.int64)
            sequence[0], sequence[1] = Keys.START, Keys.HARD_DROP
        attack, clears = commit(env, sequence, stats)

        screen.fill((0, 0, 0))
        _blit_board(screen, pygame, env)
        _blit_pieces(screen, pygame, font, env)
        _blit_distribution(screen, pygame, font, probs, best_slot)
        _blit_panel(
            screen,
            pygame,
            font,
            [
                f"turn {turn:4d}/{args.num_steps}   {last_ms:6.1f} ms/move   "
                f"{len(cand_actions):3d} candidates",
                f"attack {attack:2d}   clears {clears:2d}   "
                f"b2b {int(env._scorer._b2b):3d}   combo {int(env._scorer._combo):3d}",
                f"episode APP {stats.app:5.3f}   max b2b {stats.max_b2b:3d}   "
                f"pending garbage {int(env._get_total_garbage()):3d}",
                f"run APP {stats.run_app:5.3f}   run max b2b {stats.max_b2b_run:3d}   "
                f"deaths {stats.episodes}",
                f"oracle {args.oracle}  depth {args.search_depth}  "
                f"width {args.beam_width}   space = pause",
            ],
        )
        pygame.display.flip()
        clock.tick(args.fps)

        if env._is_top_out(env._board):
            stats.episodes += 1
            seed += 9973
            env = make_env(args, seed, traces)
            env.reset()
            oracle = build_oracle(
                args.oracle,
                env,
                args.search_depth,
                args.beam_width,
                args.queue_size,
                args.max_len,
            )
            stats.reset_episode()

    pygame.quit()


def main(args):
    """Entry point for `demo --oracle {c,gpu}`."""
    if args.oracle == "gpu":
        _quiet_low_occupancy()
    if getattr(args, "headless", False):
        run_headless(args)
    else:
        run_window(args)
