import tensorflow as tf
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.data.placement_features import build_placement_inference
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.search.placement_search import SearchConfig, search_best_move
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pygame
import numpy as np
import time

from qtris.demo.constants import PIECE_COLORS, READABLE_KEYS
from qtris.demo.panels import (
    MaxStatTracker,
    draw_bcg_panel,
    draw_board_area,
    draw_info_panel,
    draw_step_counter,
    run_replay,
)
from qtris.demo.rendering import (
    colorize_attention_scores,
    colorize_piece_sidebar,
    compute_bcg_heatmaps,
    draw_garbage_bar,
)
from qtris.demo.utils import load_checkpoint, load_piece_display, save_frames_as_video
from qtris.training.placement_az import _load_trace_pools

num_envs = 1
piece_dim = 8
depth = 64
num_heads = 4
num_layers = 4
dropout_rate = 0.0
max_len = 15
num_row_tiers = 2

# Candidate enumeration: a wide, shallow search returns every legal root
# placement (with landing rows + key sequences); the model does the ranking.
search_depth = 2
beam_width = 512

num_steps = 500
queue_size = 5
max_holes = 100
max_height = 18


def main(args):
    p_model = PlacementPolicyValueNet(
        batch_size=num_envs,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    p_model(
        (
            tf.keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            tf.keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            tf.keras.Input(shape=(3,), dtype=tf.float32),
            tf.keras.Input(shape=(None, 18), dtype=tf.float32),
            tf.keras.Input(shape=(None,), dtype=tf.bool),
        )
    )

    load_checkpoint(p_model, args.checkpoint)

    # MCTS search must use the SAME return_scale the value head was trained at - it normalizes
    # the per-edge attack / b2b / death terms against the learned value. At the default 1.0 the
    # immediate reward terms dwarf the value head (~return_scale x too large), so the search
    # cashes out / breaks b2b instead of hoarding. Restore it from the (AZ) checkpoint; BC/PPO
    # checkpoints have no return_scale -> fall back to 1.0.
    mcts_return_scale = 1.0
    try:
        _ck = tf.train.latest_checkpoint(args.checkpoint)
        _reader = tf.train.load_checkpoint(_ck)
        mcts_return_scale = float(
            _reader.get_tensor("return_scale/.ATTRIBUTES/VARIABLE_VALUE")
        )
        print(f"MCTS using trained return_scale={mcts_return_scale:.3f}", flush=True)
    except Exception:
        print("No return_scale in checkpoint; MCTS using return_scale=1.0", flush=True)

    p_model.summary()

    garbage_traces = None
    traces_dir = getattr(args, "garbage_traces", None)
    if traces_dir:
        pools = _load_trace_pools(traces_dir)
        tier = getattr(args, "trace_tier", None) or (list(pools)[-1] if pools else None)
        if tier not in pools:
            raise SystemExit(
                f"trace tier {tier!r} not found in {traces_dir} (have {list(pools)})"
            )
        garbage_traces = pools[tier]
        print(f"Trace garbage: tier {tier} ({len(garbage_traces)} traces)", flush=True)

    py_env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=num_steps,
        max_len=max_len,
        pathfinding=True,
        garbage_chance=getattr(args, "garbage_chance", 0.15),
        garbage_min=1,
        garbage_max=4,
        seed=0,
        idx=0,
        num_row_tiers=num_row_tiers,
        garbage_traces=garbage_traces,
    )
    env = TFPyEnvironment(py_env)
    searcher = CB2BSearch()
    search_cfg = (
        SearchConfig(depth=args.depth, beam_width=args.beam, gate_k=args.gate)
        if getattr(args, "search", False)
        else None
    )
    # --mcts-sims plays the AlphaZero way: PUCT MCTS over the net's policy+value picks
    # the move (greedy by visit count). No Dirichlet noise (eval, not self-play).
    mcts = (
        PlacementMCTS(
            p_model,
            MCTSConfig(
                num_simulations=args.mcts_sims,
                c_puct=args.mcts_cpuct,
                dirichlet_eps=0.0,
                leaves_per_round=getattr(args, "mcts_leaves", 4),
                gamma=1.0,
                w_attack=0.05,
                w_death=1.0,
                w_b2b=0.06,
            ),
        )
        if getattr(args, "mcts_sims", 0) > 0
        else None
    )

    screen_w = 870
    screen_h = 800
    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tetris")
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 22)

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
    value_ests = []
    max_stats = []

    death = 0
    running_attacks = 0
    running_clears = 0
    stat_tracker = MaxStatTracker()

    piece_display = load_piece_display()

    def draw_bottom_panel(surface, ind):
        draw_info_panel(
            surface,
            font,
            small_font,
            screen_w,
            [
                f"Attack Reward: {attack_rewards[ind]:0.2f}",
                f"Total Reward: {total_rewards[ind]:0.2f}",
                f"Value Est: {value_ests[ind]:0.2f}",
            ],
            [
                f"Attack: {int(attacks[ind])}",
                f"APP: {apps[ind]:0.2f}",
                f"Clear: {int(clears[ind])}",
                f"Current B2B: {current_b2b[ind]}",
                f"Current Combo: {current_combo[ind]}",
            ],
            max_stats[ind],
            actions[ind],
        )

    start = time.time()
    for t in range(num_steps):
        board = time_step.observation["board"]
        vis_board = time_step.observation.get("vis_board", None)
        b2b_combo_garbage = time_step.observation["b2b_combo_garbage"]
        pieces = time_step.observation["pieces"]
        attack = time_step.reward["attack"].numpy()[0]
        clear = time_step.reward["clear"].numpy()[0]
        attack_reward = time_step.reward["attack_reward"].numpy()[0]
        total_reward = time_step.reward["total_reward"].numpy()[0]

        current_b2b_val = py_env._scorer._b2b
        current_combo_val = py_env._scorer._combo
        current_garbage_val = py_env._get_total_garbage()
        max_stats.append(
            stat_tracker.update(current_b2b_val, current_combo_val, attack)
        )

        if time_step.is_last():
            death = t
            running_attacks = 0
            running_clears = 0
            stat_tracker.reset_episode()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Enumerate legal placements from the live search, then let the model rank.
        queue = np.array([p.value for p in py_env._queue], dtype=np.int32)
        _ba, _bs, cand_actions, cand_scores, cand_seqs, cand_rows = (
            searcher.search_with_scores(
                board=py_env._board,
                active_piece=py_env._active_piece.piece_type.value,
                hold_piece=py_env._hold_piece.value,
                queue=queue,
                b2b=int(py_env._scorer._b2b),
                combo=int(py_env._scorer._combo),
                total_garbage=int(py_env._get_total_garbage()),
                garbage_push_delay=py_env._garbage_push_delay,
                search_depth=search_depth,
                beam_width=beam_width,
                max_len=max_len,
            )
        )
        placements, cand_mask, cand_sequences = build_placement_inference(
            cand_actions,
            cand_scores,
            cand_rows,
            cand_seqs,
            active_piece=py_env._active_piece.piece_type.value,
            hold_piece=py_env._hold_piece.value,
            queue0=int(queue[0]),
            row_norm=py_env._board.shape[0] - 1,
            max_len=max_len,
        )

        key_sequence, log_prob, action_index, scores = p_model.predict(
            (
                board,
                pieces,
                b2b_combo_garbage,
                tf.constant(placements[None], dtype=tf.float32),
                tf.constant(cand_mask[None], dtype=tf.bool),
            ),
            greedy=True,
            cand_sequences=tf.constant(cand_sequences[None], dtype=tf.int64),
            temperature=1.0,
        )

        value_est = float(
            p_model.state_value(board, pieces, b2b_combo_garbage).numpy()[0, 0]
        )

        if mcts is not None:
            # AlphaZero-style play: PUCT MCTS picks the move greedily by visit count
            # (the predict above is only for the attention panel + piece_scores).
            res = mcts.search([py_env], mcts_return_scale, 0.0)[0]
            if res["dead"]:
                forced = np.full(max_len, Keys.PAD, dtype=np.int64)
                forced[0], forced[1] = Keys.START, Keys.HARD_DROP
                key_sequence = tf.constant(forced[None], dtype=tf.int64)
            else:
                # The C engine steps by descriptor; reconstruct the key sequence for the
                # chosen placement from the env pathfinder to drive the (TF-wrapped) demo env.
                is_hold, rot, norm_col, _landing, spin = res["descriptor"]
                action_index = is_hold * 160 + rot * 40 + norm_col * 4 + spin
                _, _, cand_seqs = py_env._enumerate_placement_candidates()
                key_sequence = tf.constant(
                    cand_seqs[action_index][None], dtype=tf.int64
                )
        elif search_cfg is not None:
            # Neural-guided search picks the move (the predict above is only for the
            # attention panel + piece_scores). The search handles dead states itself.
            key_sequence = tf.constant(
                search_best_move(py_env, p_model, searcher, search_cfg)[None],
                dtype=tf.int64,
            )
        elif not np.any(key_sequence.numpy()[0] == Keys.HARD_DROP):
            # No surviving placement (near-death): the env locks + scores only on a
            # HARD_DROP (else its `is_spin` is unbound), so commit a hard drop to top
            # out + auto-reset, the same death path the flat/ar demos take.
            forced = np.full(max_len, Keys.PAD, dtype=np.int64)
            forced[0], forced[1] = Keys.START, Keys.HARD_DROP
            key_sequence = tf.constant(forced[None], dtype=tf.int64)

        pieces_array = pieces.numpy()
        if pieces_array.ndim > 1:
            pieces_array = pieces_array[0]

        bcg_colored_heatmaps = compute_bcg_heatmaps(scores)
        colored_scores = colorize_attention_scores(scores, pieces_array, PIECE_COLORS)
        colored_sidebar = colorize_piece_sidebar(
            piece_display, pieces_array, PIECE_COLORS
        )
        garbage_surface = draw_garbage_bar(py_env, height=24, width=10)

        screen.fill((0, 0, 0))
        draw_board_area(
            screen,
            board,
            vis_board,
            colored_sidebar,
            colored_scores,
            garbage_surface,
            PIECE_COLORS,
        )
        draw_bcg_panel(
            screen,
            small_font,
            bcg_colored_heatmaps,
            [current_b2b_val, current_combo_val, current_garbage_val],
        )
        draw_step_counter(screen, font, t + 1, num_steps)

        readable_action = "".join(
            [READABLE_KEYS.get(k, "") for k in key_sequence.numpy()[0]]
        )

        actions.append(readable_action)

        running_attacks += attack
        running_clears += clear

        attacks.append(running_attacks)
        apps.append(running_attacks / (t - death + 1))
        clears.append(running_clears)
        attack_rewards.append(attack_reward)
        total_rewards.append(total_reward)
        value_ests.append(value_est)
        current_b2b.append(current_b2b_val)
        current_combo.append(current_combo_val)
        current_garbage.append(current_garbage_val)

        time_step = env.step(key_sequence)

        draw_bottom_panel(screen, -1)

        pygame.display.update()

        frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

    time_taken = time.time() - start

    print(f"Time taken: {time_taken:3.2f} seconds")
    print(f"Steps: {num_steps} | Time per step: {(time_taken / num_steps):1.3f}")
    save_frames_as_video(frames, "DemoPlacement.mp4")

    run_replay(screen, font, frames, num_steps, draw_bottom_panel)
