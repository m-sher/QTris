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
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
import numpy as np
import time

from qtris.demo.constants import PIECE_COLORS, READABLE_KEYS, BCG_LABELS
from qtris.demo.rendering import (
    compute_bcg_heatmaps,
    draw_garbage_bar,
    colorize_piece_sidebar,
)
from qtris.demo.utils import load_checkpoint, load_piece_display, save_frames_as_video

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

    # BCG attention panel layout
    bcg_panel_x = 680
    bcg_label_y = 5
    bcg_heatmap_y = 30
    bcg_heatmap_w = 50
    bcg_heatmap_h = 120
    bcg_gap = 15

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

    piece_display = load_piece_display()

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

        if time_step.is_last():
            death = t
            running_attacks = 0
            running_clears = 0

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

        piece_attention = tf.reduce_sum(scores, axis=[0, 2])
        # Slice out BCG tokens: keep only piece queries (:7) and patch keys (:60)
        piece_patch_attn = piece_attention[0, :7, :60]
        dominant_pieces = tf.argmax(piece_patch_attn, axis=0)
        dominant_grid = tf.reshape(dominant_pieces, (12, 5))

        dominant_attention = tf.reduce_max(piece_patch_attn, axis=0)
        dominant_attention_grid = tf.reshape(dominant_attention, (12, 5))

        bcg_colored_heatmaps = compute_bcg_heatmaps(scores)

        attention_min = tf.reduce_min(dominant_attention_grid)
        attention_max = tf.reduce_max(dominant_attention_grid)
        attention_normalized = (dominant_attention_grid - attention_min) / (
            attention_max - attention_min + 1e-8
        )

        all_PIECE_COLORS = PIECE_COLORS[pieces_array]
        colored_scores = np.zeros((12, 5, 3), dtype=np.uint8)
        dominant_grid_np = dominant_grid.numpy()
        attention_np = attention_normalized.numpy()

        for r in range(12):
            for c in range(5):
                piece_idx = dominant_grid_np[r, c]
                intensity = attention_np[r, c]
                colored_scores[r, c] = (all_PIECE_COLORS[piece_idx] * intensity).astype(
                    np.uint8
                )

        colored_sidebar = colorize_piece_sidebar(
            piece_display, pieces_array, PIECE_COLORS
        )

        garbage_surface = draw_garbage_bar(py_env, height=24, width=10)

        screen.fill((0, 0, 0))

        board_surf = pygame.Surface((10, 24))
        piece_surf = pygame.Surface((5, 28))
        scores_surf = pygame.Surface((5, 12))
        garbage_surf = pygame.Surface(
            (garbage_surface.shape[1], garbage_surface.shape[0])
        )

        if vis_board is not None:
            colored_board = PIECE_COLORS[vis_board[0, ..., 0].numpy()]
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
                f"{BCG_LABELS[i]}: {bcg_vals[i]}", True, (255, 255, 255)
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
    save_frames_as_video(frames, "DemoPlacement.mp4")

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

    Button(
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

    Button(
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
