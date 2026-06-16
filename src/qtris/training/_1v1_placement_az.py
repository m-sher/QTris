"""1v1 opponent-pool AlphaZero for the placement model.

The learner (player 1) duels an opponent (player 2) sampled each generation from a pool of
frozen past snapshots, via decoupled per-player MCTS: each player searches its own board
(the opponent's already-sent garbage is seen at the root; none is modeled landing within the
search horizon), the chosen placements are committed, and garbage is exchanged as
`PyTetris1v1Env` does. The value head regresses the realized game outcome z in {-1, 0, +1}
(loss / draw / win). The search runs at w_death=1, gamma=1, return_scale=1, w_attack=0.05,
w_b2b=0.06; own-death = -1 is the only in-search terminal.

Both players' trajectories are trained, each labeled with its own outcome z. The pool lives
on disk under `<ckpt>/pool/gen_*`; gen_0 is seeded from the warm-started net and is the frozen
reference for the periodic `win_rate_vs_ref` eval. Opponents are sampled recency-weighted per
generation; the pool grows (gated on the learner's decisive win-rate EMA) and evicts oldest
(gen_0 pinned).
"""

import glob
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.observability.backend import finish, init_run, log_step
from qtris.observability.models import OneVsOneAZLog, OneVsOnePlacementAZConfig
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.search.placement_search import placement_step
from qtris.training.placement_az import _gen_log_probs, train_step


def _build_game_pairs(num_games, queue_size, max_holes, max_height, max_len, seed0=123):
    """List of (env1, env2) raw PyTetrisEnv pairs built like PyTetris1v1Env's sub-envs:
    no random garbage, manual garbage push + queue fill, no env step cap (the loop caps
    games). Both envs of a pair share a seed (mirror-fair pieces); games differ."""
    pairs = []
    for g in range(num_games):
        kw = dict(
            queue_size=queue_size,
            max_holes=max_holes,
            max_height=max_height,
            max_steps=None,
            max_len=max_len,
            pathfinding=False,
            seed=seed0 + g,
            idx=g,
            garbage_chance=0.0,
            garbage_min=0,
            garbage_max=0,
            auto_push_garbage=False,
            auto_fill_queue=False,
            num_row_tiers=2,
            placement_candidates=False,
        )
        pairs.append((PyTetrisEnv(**kw), PyTetrisEnv(**kw)))
    return pairs


def _pos(r):
    """Storable position from one player's MCTS result dict."""
    return {
        "board": r["board"],
        "pieces": r["pieces"],
        "bcg": r["bcg"],
        "cand_placements": r["cand_placements"],
        "cand_mask": r["cand_mask"],
        "pi": r["pi"],
        "v_root": r["value"],
        "visits": r["visits"],
    }


def _commit_and_exchange(env1, env2, searcher, desc1, desc2, rng):
    """Commit both pre-chosen placements, then replicate PyTetris1v1Env._step's garbage
    exchange / push timing / death logic on the raw sub-envs (PyTetris1v1Env.py:151-249).
    Returns (p1_died, p2_died, attack1, attack2)."""
    # placement_step already cancels each player's own pending garbage and (auto_push_garbage
    # =False) does NOT push to the board, so net = attack - cancelled is the queue delta.
    info = []
    for env, desc in ((env1, desc1), (env2, desc2)):
        pre_b2b = env._scorer._b2b
        pending_before = env._get_total_garbage()
        _total, attack, clears, died = placement_step(env, searcher, desc)
        pending_after = env._get_total_garbage()
        net = attack - (pending_before - pending_after)
        # Actual surge = a b2b chain (>=4) broken by this clear (releases banked b2b).
        is_surge = clears > 0 and pre_b2b >= 4 and env._scorer._b2b == -1
        info.append(
            {
                "died": died,
                "attack": attack,
                "clears": clears,
                "net": net,
                "surge": is_surge,
            }
        )
    # Push existing garbage for non-clearing players (so incoming sits >=1 turn first).
    for env, i in ((env1, info[0]), (env2, info[1])):
        if i["clears"] == 0 and env._garbage_queue:
            env._tick_garbage_timers()
            env._board, env._vis_board, _ = env._push_garbage_to_board(
                env._board, env._vis_board
            )
    # Inject net attacks into the opponent (real surges split into waves); lands next turn.
    if info[0]["net"] > 0:
        env2._receive_attack(int(info[0]["net"]), rng.randint(0, 9), info[0]["surge"])
    if info[1]["net"] > 0:
        env1._receive_attack(int(info[1]["net"]), rng.randint(0, 9), info[1]["surge"])
    # Death = own-placement death, garbage-induced top-out (re-check after the push), or holes.
    died = []
    for env, i in ((env1, info[0]), (env2, info[1])):
        d = i["died"] or bool(np.any(env._board[: 24 - env._max_height] != 0.0))
        if env._max_holes is not None:
            _h, holes, _s, _b = env._board_stats(env._board)
            d = d or holes > env._max_holes
        died.append(d)
    # Refill queues (sub-envs are auto_fill_queue=False; placement_step skipped this).
    env1._queue = env1._fill_queue(env1._queue)
    env2._queue = env2._fill_queue(env2._queue)
    return died[0], died[1], info[0]["attack"], info[1]["attack"]


def _episode(pend, p1_died, p2_died):
    """Stamp each player's realized outcome z on its pending positions and return both
    players' rows for training. Returns (rows[(pos, z)], game_len, p1_won, is_draw) keyed on
    the learner's (player-1) outcome, or None if the game collected nothing."""
    glen = max(len(pend["p1"]), len(pend["p2"]))
    if glen == 0:
        return None
    if p1_died and not p2_died:
        z1, z2 = -1.0, 1.0
    elif p2_died and not p1_died:
        z1, z2 = 1.0, -1.0
    else:
        z1, z2 = 0.0, 0.0
    # Third element is the policy mask: the learner's (p1) positions train the policy;
    # the opponent's (p2) positions train the value only (avoids distilling a frozen policy).
    rows = [(p, z1, 1.0) for p in pend["p1"]] + [(p, z2, 0.0) for p in pend["p2"]]
    return rows, glen, z1 > 0.0, z1 == 0.0


def _build_net(batch_size, piece_dim, depth, num_heads, num_layers, queue_size):
    """A tanh-value PlacementPolicyValueNet with its variables built (ready for restore)."""
    net = PlacementPolicyValueNet(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=0.0,
        value_activation="tanh",  # bound the value to the outcome target's [-1, 1]
    )
    net(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    return net


def _pool_snaps(pool_dir):
    """Pool snapshot prefixes, sorted ascending by generation number."""
    idx = glob.glob(os.path.join(pool_dir, "gen_*.index"))
    snaps = [f[: -len(".index")] for f in idx]
    return sorted(snaps, key=lambda p: int(os.path.basename(p).split("_")[1]))


def _save_pool(net, gen, pool_dir, max_pool_size):
    """Snapshot the learner's weights into the pool, then FIFO-evict oldest (gen_0 pinned)."""
    os.makedirs(pool_dir, exist_ok=True)
    net.save_weights(os.path.join(pool_dir, f"gen_{gen}"))
    snaps = _pool_snaps(pool_dir)
    while len(snaps) > max_pool_size:
        # Pin gen_0; evict the next-oldest snapshot.
        victim = snaps[1] if os.path.basename(snaps[0]) == "gen_0" else snaps[0]
        for f in glob.glob(victim + ".*"):
            os.remove(f)
        snaps = _pool_snaps(pool_dir)


def _sample_pool(opp_net, pool_dir):
    """Load a recency-weighted opponent into opp_net; return its gen tag (or None if empty)."""
    snaps = _pool_snaps(pool_dir)
    if not snaps:
        return None
    weights = list(range(1, len(snaps) + 1))  # newest weighted highest
    chosen = random.choices(snaps, weights=weights, k=1)[0]
    opp_net.load_weights(chosen)
    return os.path.basename(chosen)


def _eval_vs_ref(
    learner_mcts, ref_mcts, n_games, queue_size, max_len, max_steps, rng, searcher
):
    """Decisive win rate of the learner (player 1) vs the frozen reference (player 2), both
    greedy, played to completion on fresh games. Batched over still-live games each round."""
    pairs = _build_game_pairs(n_games, queue_size, 50, 18, max_len, seed0=9001)
    for e1, e2 in pairs:
        e1._reset()
        e2._reset()
    alive = [True] * n_games
    mc = np.zeros(n_games, dtype=np.int64)
    wins = losses = 0
    for _t in range(max_steps):
        idx = [g for g in range(n_games) if alive[g]]
        if not idx:
            break
        temps = np.zeros(len(idx), dtype=np.float32)  # greedy both sides
        r1 = learner_mcts.search([pairs[g][0] for g in idx], 1.0, temps)
        r2 = ref_mcts.search([pairs[g][1] for g in idx], 1.0, temps)
        for j, g in enumerate(idx):
            a, b = r1[j], r2[j]
            if a["dead"] or b["dead"]:
                wins += int(b["dead"] and not a["dead"])
                losses += int(a["dead"] and not b["dead"])
                alive[g] = False
                continue
            p1_died, p2_died, _a1, _a2 = _commit_and_exchange(
                pairs[g][0],
                pairs[g][1],
                searcher,
                a["descriptor"],
                b["descriptor"],
                rng,
            )
            mc[g] += 1
            if p1_died or p2_died:
                wins += int(p2_died and not p1_died)
                losses += int(p1_died and not p2_died)
                alive[g] = False
            elif mc[g] >= max_steps:
                alive[g] = False  # timeout = draw, excluded from decisive WR
    dec = wins + losses
    return wins / dec if dec else 0.5


def main(args):
    piece_dim, depth, num_heads, num_layers = 8, 64, 4, 4
    queue_size, max_len = 5, 15
    num_games = getattr(args, "num_games", 16)
    horizon = getattr(args, "horizon", 32)
    max_game_steps = getattr(args, "max_game_steps", 512)
    num_generations = getattr(args, "num_generations", 1_000_000)
    mini_batch_size = getattr(args, "mini_batch_size", 256)
    num_epochs = getattr(args, "num_epochs", 2)
    value_coef = getattr(args, "value_coef", 1.0)
    learning_rate = getattr(args, "learning_rate", 1e-4)
    replay_capacity = getattr(args, "replay_capacity", 25_000)
    # Opponent-pool knobs.
    max_pool_size = getattr(args, "max_pool_size", 30)
    pool_interval = getattr(args, "pool_interval", 10)
    pool_wr_gate = getattr(args, "pool_wr_gate", 0.55)
    eval_interval = getattr(args, "eval_interval", 10)
    eval_games = getattr(args, "eval_games", 8)
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints/placement_az")
    if checkpoint_dir == "checkpoints/placement_az":
        checkpoint_dir = "checkpoints/1v1_placement_az"
    pool_dir = os.path.join(checkpoint_dir, "pool")
    run_name = getattr(args, "run_name", None)
    np_seed = getattr(args, "np_seed", None)

    if np_seed is not None:
        np.random.seed(np_seed)
    rng = random.Random(np_seed if np_seed is not None else 0)

    # Outcome-z value target; search reward = small attack credit + b2b-build shaping,
    # own-death = -1, undiscounted, scale 1.
    cfg = MCTSConfig(
        num_simulations=getattr(args, "num_simulations", 256),
        c_puct=getattr(args, "c_puct", 1.5),
        dirichlet_alpha=getattr(args, "dirichlet_alpha", 0.3),
        dirichlet_eps=getattr(args, "dirichlet_eps", 0.25),
        gamma=1.0,
        temp_moves=getattr(args, "temp_moves", 12),
        w_attack=0.05,
        w_death=1.0,
        w_b2b=getattr(args, "w_b2b", 0.06),
        leaves_per_round=getattr(args, "leaves_per_round", 4),
        vloss=getattr(args, "vloss", 1.0),
    )

    # Learner (player 1, trained); opponent + reference are frozen snapshots.
    net = _build_net(num_games, piece_dim, depth, num_heads, num_layers, queue_size)
    optimizer = keras.optimizers.Adam(learning_rate, clipnorm=0.5)
    net.compile(optimizer=optimizer, jit_compile=True)
    net.summary()
    opp_net = _build_net(num_games, piece_dim, depth, num_heads, num_layers, queue_size)
    ref_net = _build_net(
        eval_games, piece_dim, depth, num_heads, num_layers, queue_size
    )

    checkpoint = tf.train.Checkpoint(model=net, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Resumed 1v1 AZ checkpoint {manager.latest_checkpoint}.", flush=True)
    else:
        warm = tf.train.latest_checkpoint("checkpoints/placement_pretrained_policy")
        if warm is not None:
            # Policy warm-start only; the value head restores partial (left fresh).
            tf.train.Checkpoint(model=net).restore(warm).expect_partial()
            print(f"Warm-started policy from BC checkpoint {warm}.", flush=True)

    # Seed the pool with gen_0 = the warm-started learner.
    if not _pool_snaps(pool_dir):
        _save_pool(net, 0, pool_dir, max_pool_size)
        print(f"Seeded opponent pool gen_0 at {pool_dir}.", flush=True)
    # Reference net = frozen gen_0, used by the win_rate_vs_ref eval.
    ref_prefix = os.path.join(pool_dir, "gen_0")
    ref_net.load_weights(ref_prefix)

    resumed = manager.latest_checkpoint is not None
    config = OneVsOnePlacementAZConfig(
        num_games=num_games,
        horizon=horizon,
        max_game_steps=max_game_steps,
        num_simulations=cfg.num_simulations,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        temp_moves=cfg.temp_moves,
        w_b2b=cfg.w_b2b,
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        value_coef=value_coef,
        learning_rate=learning_rate,
        replay_capacity=replay_capacity,
        max_pool_size=max_pool_size,
        pool_interval=pool_interval,
        pool_wr_gate=pool_wr_gate,
        eval_interval=eval_interval,
        eval_games=eval_games,
        resumed=resumed,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        np_seed=np_seed,
    )
    run = init_run(
        project="Tetris",
        config=config,
        wandb_mirror=getattr(args, "wandb", False),
        run_name=run_name,
    )

    pairs = _build_game_pairs(num_games, queue_size, 50, 18, max_len)
    mcts = PlacementMCTS(net, cfg)
    opp_mcts = PlacementMCTS(opp_net, cfg)
    ref_mcts = PlacementMCTS(ref_net, cfg)
    searcher = (
        CB2BSearch()
    )  # lock-score core for committing the chosen move by descriptor

    for e1, e2 in pairs:
        e1._reset()
        e2._reset()
    move_count = np.zeros(num_games, dtype=np.int64)
    # Per-game pending positions for BOTH players, carried across gens until the game ends.
    pending = [{"p1": [], "p2": []} for _ in range(num_games)]

    replay = deque()
    replay_size = 0
    N = num_games
    opp_temps = np.zeros(N, dtype=np.float32)  # greedy move selection for the opponent
    wr_ema = 0.5
    last_wr_ref = 0.5

    for gen in range(num_generations):
        opp_tag = _sample_pool(opp_net, pool_dir)  # this generation's adversary

        gen_pos = []  # (pos, z) for both players' positions whose game completed this gen
        game_lens, p1_wins = [], []  # p1_wins: one bool per DECISIVE game
        n_draw = 0
        total_attack = total_placements = 0
        dead_searches = total_searches = 0

        for _t in range(horizon):
            temps_p1 = np.where(move_count < cfg.temp_moves, 1.0, 0.0).astype(
                np.float32
            )
            r1 = mcts.search([p[0] for p in pairs], 1.0, temps_p1)  # learner
            r2 = opp_mcts.search([p[1] for p in pairs], 1.0, opp_temps)  # pool opponent
            total_searches += N
            dead_searches += sum(1 for g in range(N) if r1[g]["dead"])

            for g in range(N):
                a, b = r1[g], r2[g]
                e1, e2 = pairs[g]

                if a["dead"] or b["dead"]:
                    ep = _episode(pending[g], a["dead"], b["dead"])
                else:
                    pending[g]["p1"].append(_pos(a))
                    pending[g]["p2"].append(_pos(b))
                    p1_died, p2_died, atk1, atk2 = _commit_and_exchange(
                        e1, e2, searcher, a["descriptor"], b["descriptor"], rng
                    )
                    total_attack += atk1 + atk2
                    total_placements += 2
                    move_count[g] += 1
                    cap = move_count[g] >= max_game_steps
                    if not (p1_died or p2_died or cap):
                        continue
                    ep = _episode(pending[g], p1_died, p2_died)

                if ep is not None:
                    rows, glen, p1_won, draw = ep
                    gen_pos.extend(rows)
                    game_lens.append(glen)
                    if draw:
                        n_draw += 1
                    else:
                        p1_wins.append(p1_won)
                e1._reset()
                e2._reset()
                move_count[g] = 0
                pending[g] = {"p1": [], "p2": []}

        n_new = len(gen_pos)
        if n_new == 0:
            print(f"Gen {gen}: no games completed; skipping update.", flush=True)
            continue

        boards = np.stack([p["board"] for p, _z, _m in gen_pos]).astype(np.float32)
        pieces = np.stack([p["pieces"] for p, _z, _m in gen_pos]).astype(np.int64)
        bcg = np.stack([p["bcg"] for p, _z, _m in gen_pos]).astype(np.float32)
        cand_pl = np.stack([p["cand_placements"] for p, _z, _m in gen_pos]).astype(
            np.float32
        )
        cand_mk = np.stack([p["cand_mask"] for p, _z, _m in gen_pos]).astype(bool)
        pi_tgt = np.stack([p["pi"] for p, _z, _m in gen_pos]).astype(np.float32)
        value_tgt = np.array([z for _p, z, _m in gen_pos], dtype=np.float32)
        policy_mask = np.array([m for _p, _z, m in gen_pos], dtype=np.float32)
        v_root = np.array([p["v_root"] for p, _z, _m in gen_pos], dtype=np.float32)
        visits = np.array([p["visits"] for p, _z, _m in gen_pos], dtype=np.float32)

        replay.append(
            {
                "boards": boards,
                "pieces": pieces,
                "bcg": bcg,
                "cand_placements": cand_pl,
                "cand_mask": cand_mk,
                "pi_target": pi_tgt,
                "value_target": value_tgt,
                "policy_mask": policy_mask,
            }
        )
        replay_size += n_new
        while replay_size > replay_capacity and len(replay) > 1:
            replay_size -= len(replay.popleft()["value_target"])

        if replay_size < mini_batch_size:
            print(
                f"Gen {gen}: replay {replay_size} < batch {mini_batch_size}; skipping update.",
                flush=True,
            )
            continue

        full = {k: np.concatenate([e[k] for e in replay], axis=0) for k in replay[0]}
        total_steps = num_epochs * max(1, n_new // mini_batch_size)
        ds = (
            tf.data.Dataset.from_tensor_slices(full)
            .shuffle(replay_size)
            .repeat()
            .batch(mini_batch_size, drop_remainder=True)
            .take(total_steps)
            .prefetch(tf.data.AUTOTUNE)
        )

        # update_kl over a fixed-size slice of this gen's new LEARNER positions (one trace).
        learner_idx = np.flatnonzero(policy_mask == 1.0)[:mini_batch_size]
        measure_kl = len(learner_idx) >= mini_batch_size
        if measure_kl:
            gi = (
                tf.constant(boards[learner_idx]),
                tf.constant(pieces[learner_idx]),
                tf.constant(bcg[learner_idx]),
                tf.constant(cand_pl[learner_idx]),
                tf.constant(cand_mk[learner_idx]),
            )
            lp_before = _gen_log_probs(net, *gi).numpy()

        updates = 0
        step_out = None
        for batch in ds:
            step_out = train_step(net, batch, tf.constant(value_coef, tf.float32))
            updates += 1
        if step_out is None:
            print(f"Gen {gen}: no batch produced; skipping update.", flush=True)
            continue

        if measure_kl:
            lp_after = _gen_log_probs(net, *gi).numpy()
            update_kl = float(
                (np.exp(lp_before) * (lp_before - lp_after)).sum(axis=-1).mean()
            )
        else:
            update_kl = 0.0

        n_games = len(game_lens)
        decisive = len(p1_wins)
        win_rate = float(np.mean(p1_wins)) if decisive else 0.0
        draw_rate = n_draw / n_games if n_games else 0.0
        app = total_attack / total_placements if total_placements else 0.0
        dec = (value_tgt != 0.0) & (policy_mask == 1.0)  # learner positions only
        if (
            dec.sum() >= 2
            and np.std(v_root[dec]) > 1e-6
            and np.std(value_tgt[dec]) > 1e-6
        ):
            value_calibration = float(np.corrcoef(v_root[dec], value_tgt[dec])[0, 1])
        else:
            value_calibration = 0.0

        # Pool maintenance: EMA the decisive WR, grow the pool (gated), and periodically
        # eval vs the frozen gen_0 reference.
        if decisive > 0:
            wr_ema = 0.9 * wr_ema + 0.1 * win_rate
        if gen % eval_interval == 0:
            last_wr_ref = _eval_vs_ref(
                mcts,
                ref_mcts,
                eval_games,
                queue_size,
                max_len,
                max_game_steps,
                rng,
                searcher,
            )
        if (
            gen > 0
            and gen % pool_interval == 0
            and decisive >= 8
            and wr_ema >= pool_wr_gate
        ):
            _save_pool(net, gen, pool_dir, max_pool_size)
            print(f"Saved opponent-pool gen_{gen} (wr_ema {wr_ema:.3f}).", flush=True)

        log_step(
            OneVsOneAZLog(
                policy_loss=step_out["policy_loss"],
                value_loss=step_out["value_loss"],
                entropy=step_out["entropy"],
                policy_kl=step_out["policy_kl"],
                update_kl=update_kl,
                explained_var=step_out["explained_var"],
                value_mean=step_out["value_mean"],
                avg_game_len=float(np.mean(game_lens)),
                win_rate=win_rate,
                win_rate_vs_ref=last_wr_ref,
                draw_rate=draw_rate,
                app=app,
                value_calibration=value_calibration,
                avg_b2b=float(bcg[:, 0].mean()),
                max_b2b=float(bcg[:, 0].max()),
                avg_combo=float(bcg[:, 1].mean()),
                surge_rate=float((bcg[:, 0] >= 4).mean()),
                avg_visits=float(visits.mean()),
                dead_rate=dead_searches / total_searches if total_searches else 0.0,
                updates=updates,
                buffer_size=replay_size,
                completed_games=n_games,
                pool_size=len(_pool_snaps(pool_dir)),
                board=batch["boards"][0, ..., 0].numpy(),
            )
        )
        print(
            f"Gen {gen} | Policy: {float(step_out['policy_loss']):2.3f} | "
            f"Value: {float(step_out['value_loss']):2.3f} | "
            f"Ent: {float(step_out['entropy']):1.3f} | "
            f"WR(pool {opp_tag}): {win_rate:1.2f} | WRvsRef: {last_wr_ref:1.2f} | "
            f"Games: {n_games} | APP: {app:1.3f} | Updates: {updates}",
            flush=True,
        )

        if gen % 5 == 0:
            manager.save()

    finish(run)
