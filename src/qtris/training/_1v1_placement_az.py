"""1v1 productive-attack learning with a B2B own-death risk gate."""

import glob
import os
import random
import time
from pathlib import Path
from collections import deque
from dataclasses import replace as dc_replace

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.dagger import _state_record, save_states
from qtris.data.placement_features import (
    MCTS_CANDIDATE_CAPACITY as CANDIDATE_CAPACITY,
    PLACEMENT_FEATURE_DIM,
)
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.observability.backend import finish, init_run, log_step
from qtris.observability.models import (
    OneVsOneAZLog,
    OneVsOnePlacementAZConfig,
    OneVsOneCollectionLog,
)
from qtris.search.placement_mcts import PlacementMCTS
from qtris.search.placement_search import placement_step
from qtris.training.whr import WHRBook
from qtris.training.placement_az import _gen_log_probs, warm_start_policy_only
from qtris.training.attack_risk import (
    OBJECTIVE,
    PROFILE_NAME,
    attack_risk_config,
    death_targets,
    learner_values,
    n_step_attack,
    prepare_destination,
    risk_calibration,
    save_checkpoint,
    save_profile,
    state_observation,
    train_step,
)
from qtris.search.cmcts import RISK_HORIZON


def _build_game_pairs(num_games, queue_size, max_holes, max_len, seed0=123):
    """List of (env1, env2) raw PyTetrisEnv pairs built like PyTetris1v1Env's sub-envs:
    no random garbage, manual garbage push + queue fill, no env step cap (the loop caps
    games). Both envs of a pair share a seed (mirror-fair pieces); games differ."""
    pairs = []
    for g in range(num_games):
        kw = dict(
            queue_size=queue_size,
            max_holes=max_holes,
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
        "slot": r["slot"],
        "risk_prediction": r["risk_prediction"][r["slot"]],
        "risk_curve": r["risk_curve"],
    }


def _commit_and_exchange(env1, env2, searcher, desc1, desc2, rng):
    """Commit both pre-chosen placements, then replicate PyTetris1v1Env._step's garbage
    exchange / push timing / death logic on the raw sub-envs.
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
        d = i["died"] or env._is_top_out(env._board)
        if env._max_holes is not None:
            _h, holes, _s, _b = env._board_stats(env._board)
            d = d or holes > env._max_holes
        died.append(d)
    # Refill queues (sub-envs are auto_fill_queue=False; placement_step skipped this).
    env1._queue = env1._fill_queue(env1._queue)
    env2._queue = env2._fill_queue(env2._queue)
    return died[0], died[1], info[0]["attack"], info[1]["attack"]


def _mean_or_none(xs):
    """Mean of a per-generation sample, or None when the generation had no events."""
    return float(np.mean(xs)) if xs else None


def _collection_metrics(
    search_metrics,
    learner_attack,
    productive_attack,
    surge_attack,
    learner_placements,
    elapsed,
):
    return {
        "attack/total_app": learner_attack / max(1, learner_placements),
        "attack/productive_app": productive_attack / max(1, learner_placements),
        "attack/surge_app": surge_attack / max(1, learner_placements),
        "progress/generation_seconds": elapsed,
        "search/max_depth": _mean_or_none([r["max_depth"] for r in search_metrics]),
        "search/eligible_candidates": _mean_or_none(
            [int(r["gate_mask"].sum()) for r in search_metrics]
        ),
        "gate/breaks_chosen": sum(
            int(r["break_mask"][r["slot"]]) for r in search_metrics
        ),
        "gate/break_risk_reduction": _mean_or_none(
            [
                r["risk_best_keep"] - r["risk_chosen"]
                for r in search_metrics
                if r["break_mask"][r["slot"]] and r["risk_best_keep"] is not None
            ]
        ),
    }


def _episode(pend, p1_died, p2_died, n_step, gamma=0.97, tails=(0.0, 0.0)):
    """Stamp productive-attack and censored own-death targets on both players' rows."""
    glen = max(len(pend["p1"]), len(pend["p2"]))
    if not glen:
        return None
    z1 = float(p2_died) - float(p1_died)
    rows = []
    for player, died, z, pm, tail in (
        ("p1", p1_died, z1, 1.0, tails[0]),
        ("p2", p2_died, -z1, 0.0, tails[1]),
    ):
        positions = pend[player]
        targets = n_step_attack(
            [p["reward"] for p in positions],
            [p["bootstrap_value"] for p in positions],
            n_step,
            gamma,
            0.0 if died else tail,
        )
        hazards, mask, cumulative, observed = death_targets(len(positions), died)
        for i, (pos, target) in enumerate(zip(positions, targets)):
            pos.update(
                hazard_target=hazards[i],
                hazard_mask=mask[i],
                death_target=cumulative[i],
                death_observed=observed[i],
                own_terminal=bool(died and i == len(positions) - 1),
            )
            rows.append((pos, target, pm, z, len(positions) - i - 1))
    return rows, glen, z1 > 0, z1 == 0


def _finalize_episodes(episodes, net, batch_size, n_step, gamma):
    """Evaluate all completed trajectories with the same pre-update learner."""
    observations = []
    for pend, _d1, _d2, tails in episodes:
        for player in ("p1", "p2"):
            observations.extend(
                (p["board"], p["pieces"], p["bcg"]) for p in pend[player]
            )
        observations.extend(tails)
    values = iter(learner_values(net, observations, batch_size))
    completed = []
    for pend, died1, died2, _tails in episodes:
        for player in ("p1", "p2"):
            for pos in pend[player]:
                pos["bootstrap_value"] = float(next(values))
        tails = (float(next(values)), float(next(values)))
        ep = _episode(pend, died1, died2, n_step, gamma, tails)
        if ep is not None:
            completed.append(ep)
    return completed


def _build_net(
    batch_size, piece_dim, depth, num_heads, num_layers, queue_size, attack_risk=False
):
    """Build the selected placement critic profile and its restore-ready variables."""
    net = PlacementPolicyValueNet(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=0.0,
        value_activation=None if attack_risk else "tanh",
        risk_horizon=RISK_HORIZON if attack_risk else 0,
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


def _save_pool(net, gen, pool_dir, max_pool_size, cfg=None, n_step=14):
    """Snapshot the learner's weights into the pool, then FIFO-evict oldest (gen_0 pinned)."""
    os.makedirs(pool_dir, exist_ok=True)
    prefix = os.path.join(pool_dir, f"gen_{gen}")
    if os.path.exists(prefix + ".index"):
        # Overwriting would silently corrupt the WHR log's gen_k = learner-at-k identity.
        raise FileExistsError(f"pool snapshot {prefix} already exists")
    net.save_weights(prefix)
    if cfg is not None:
        save_profile(prefix + ".objective.json", cfg, n_step)
    snaps = _pool_snaps(pool_dir)
    while len(snaps) > max_pool_size:
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
    opp_net.load_weights(chosen).expect_partial()
    return os.path.basename(chosen)


def _eval_vs_ref(
    learner_mcts,
    ref_mcts,
    n_games,
    queue_size,
    max_len,
    max_steps,
    rng,
    searcher,
    seed0,
):
    """(wins, losses, draws) of the learner (player 1) vs the frozen reference (player 2),
    both greedy, played to completion on fresh games. Batched over still-live games each
    round. Draws (timeouts + double-KOs) enter the rating fit as half-wins.

    Games use seeds seed0..seed0+n_games-1, so `seed0` must stride by at least n_games
    between calls."""
    pairs = _build_game_pairs(n_games, queue_size, 50, max_len, seed0=seed0)
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
                alive[g] = False  # timeout = draw
    return wins, losses, n_games - wins - losses


def main(args):
    piece_dim, depth, num_heads, num_layers = 8, 64, 4, 4
    queue_size, max_len = 5, 15
    num_games = getattr(args, "num_games", 16)
    horizon = getattr(args, "horizon", 32)
    max_game_steps = getattr(args, "max_game_steps", 512)
    num_generations = getattr(args, "num_generations", 1_000_000)
    mini_batch_size = getattr(args, "batch_size", 256)
    num_epochs = getattr(args, "num_epochs", 2)
    value_coef = getattr(args, "value_coef", 1.0)
    risk_coef = getattr(args, "risk_coef", 1.0)
    learning_rate = getattr(args, "learning_rate", 3e-4)
    replay_capacity = getattr(args, "replay_capacity", 8_000)
    # Opponent-pool knobs.
    max_pool_size = getattr(args, "max_pool_size", 30)
    pool_interval = getattr(args, "pool_interval", 10)
    pool_wr_gate = getattr(args, "pool_wr_gate", 0.55)
    eval_interval = getattr(args, "eval_interval", 20)
    eval_games = getattr(args, "eval_games", 32)
    n_step = max(1, int(getattr(args, "n_step", 14)))
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints/placement_az")
    if checkpoint_dir == "checkpoints/placement_az":
        checkpoint_dir = "checkpoints/1v1_attack_risk"
    pool_dir = os.path.join(checkpoint_dir, "pool")
    run_name = getattr(args, "run_name", None)
    seed = getattr(args, "seed", None)
    save_states_dir = getattr(args, "save_states", None)
    # Opponent-pool rating (WHR) knobs.
    elo_enabled = getattr(args, "elo_enabled", True)
    elo_init = getattr(args, "elo_init", 1500.0)
    whr_drift = getattr(args, "whr_drift", 8.0)
    whr_tie_sigma = getattr(args, "whr_tie_sigma", 70.0)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
    rng = random.Random(seed if seed is not None else 0)

    cfg = attack_risk_config(
        num_simulations=getattr(args, "num_simulations", 256),
        c_puct=getattr(args, "c_puct", 1.5),
        dirichlet_alpha=getattr(args, "dirichlet_alpha", 0.3),
        dirichlet_eps=getattr(args, "dirichlet_eps", 0.25),
        gamma=0.97 if getattr(args, "gamma", None) is None else args.gamma,
        temp_moves=getattr(args, "temp_moves", 12),
        risk_threshold=getattr(args, "risk_threshold", 0.10),
        risk_margin=getattr(args, "risk_margin", 0.05),
        q_norm=bool(getattr(args, "q_norm", True)),
        leaves_per_round=getattr(args, "leaves_per_round", 4),
        vloss=getattr(args, "vloss", 1.0),
    )

    if (
        not 0 < cfg.gamma <= 1
        or not 0 <= cfg.risk_threshold <= 1
        or not 0 <= cfg.risk_margin <= 1
    ):
        raise ValueError("invalid discount or risk gate probabilities")
    init_source = prepare_destination(
        checkpoint_dir, getattr(args, "init_checkpoint", None), cfg, n_step
    )
    save_profile(Path(pool_dir) / PROFILE_NAME, cfg, n_step)

    # Learner (player 1, trained); opponent + reference are frozen snapshots.
    net = _build_net(
        num_games, piece_dim, depth, num_heads, num_layers, queue_size, attack_risk=True
    )
    optimizer = keras.optimizers.Adam(learning_rate, clipnorm=0.5)
    net.compile(optimizer=optimizer, jit_compile=True)
    net.summary()
    opp_net = _build_net(
        num_games, piece_dim, depth, num_heads, num_layers, queue_size, attack_risk=True
    )
    ref_net = _build_net(
        eval_games,
        piece_dim,
        depth,
        num_heads,
        num_layers,
        queue_size,
        attack_risk=True,
    )

    next_generation = tf.Variable(0, dtype=tf.int64, trainable=False)
    optimizer.build(net.trainable_variables)
    checkpoint = tf.train.Checkpoint(
        model=net, optimizer=optimizer, next_generation=next_generation
    )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()
        print(f"Resumed 1v1 AZ checkpoint {manager.latest_checkpoint}.", flush=True)
    else:
        warm = init_source or tf.train.latest_checkpoint(
            "checkpoints/placement_pretrained_policy"
        )
        if warm is not None:
            warm_start_policy_only(net, warm)
            print(
                f"Initialized encoder/policy from {warm}; critics and optimizer fresh.",
                flush=True,
            )

    # Seed the pool with gen_0 = the warm-started learner.
    if not _pool_snaps(pool_dir):
        _save_pool(net, 0, pool_dir, max_pool_size, cfg, n_step)
        print(f"Seeded opponent pool gen_0 at {pool_dir}.", flush=True)
    # Reference net = frozen gen_0, used by the win_rate_vs_ref eval.
    ref_prefix = os.path.join(pool_dir, "gen_0")
    ref_net.load_weights(ref_prefix).expect_partial()

    # Opponent-pool WHR book (batch refit over pool/games.jsonl, anchored at gen_0).
    games_path = os.path.join(pool_dir, "games.jsonl")
    ratings_path = os.path.join(pool_dir, "ratings.json")
    legacy_elo = os.path.join(pool_dir, "elo.json")
    if os.path.exists(legacy_elo) and not os.path.exists(games_path):
        os.remove(legacy_elo)
        print("Migrated to WHR ratings: removed legacy elo.json.", flush=True)
    whr = None
    if elo_enabled:
        whr = WHRBook(
            games_path,
            pool_dir=pool_dir,
            init=elo_init,
            drift=whr_drift,
            tie_sigma=whr_tie_sigma,
        )
        if whr.last_gen >= 0:
            print(f"Resumed WHR log from {games_path}.", flush=True)

    # Resume-safe monotone generation: the append-only log and gen_k snapshot ids
    # both key on it, so it must never restart at 0 (gen_0 seed excluded).
    gen0 = int(next_generation.numpy())
    if whr is not None:
        gen0 = max(gen0, whr.last_gen + 1)
    for snap in _pool_snaps(pool_dir):
        k = int(os.path.basename(snap).split("_")[1])
        if k > 0:
            gen0 = max(gen0, k + 1)
    if gen0 > 0:
        print(f"Resuming at global generation {gen0}.", flush=True)

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
        w_attack=cfg.w_attack,
        w_b2b=cfg.w_b2b,
        w_height=cfg.w_height,
        w_bumpiness=cfg.w_bumpiness,
        q_norm=cfg.q_norm,
        fpu=cfg.fpu,
        w_holes=cfg.w_holes,
        w_plain=cfg.w_plain,
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
        n_step=n_step,
        objective=OBJECTIVE,
        gamma=cfg.gamma,
        risk_horizon=RISK_HORIZON,
        risk_threshold=cfg.risk_threshold,
        risk_margin=cfg.risk_margin,
        risk_coef=risk_coef,
        leaves_per_round=cfg.leaves_per_round,
        vloss=cfg.vloss,
        w_death=cfg.w_death,
        init_checkpoint=init_source,
        resumed=resumed,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        seed=seed,
        save_states=save_states_dir,
        elo_enabled=elo_enabled,
        elo_init=elo_init,
        whr_drift=whr_drift,
        whr_tie_sigma=whr_tie_sigma,
    )
    run = init_run(
        project="Tetris",
        config=config,
        wandb_mirror=getattr(args, "wandb", False),
        run_name=run_name,
    )

    pairs = _build_game_pairs(
        num_games, queue_size, 50, max_len, seed0=seed if seed is not None else 123
    )
    mcts = PlacementMCTS(net, cfg)
    opp_mcts = PlacementMCTS(opp_net, cfg)
    eval_cfg = dc_replace(cfg, dirichlet_eps=0.0)
    eval_mcts = PlacementMCTS(net, eval_cfg)
    ref_mcts = PlacementMCTS(ref_net, eval_cfg)
    searcher = (
        CB2BSearch()
    )  # lock-score core for committing the chosen move by descriptor

    for e1, e2 in pairs:
        e1._reset()
        e2._reset()
    move_count = np.zeros(num_games, dtype=np.int64)
    # Per-game pending positions for BOTH players, carried across gens until the game ends.
    pending = [{"p1": [], "p2": []} for _ in range(num_games)]
    cur_chain = [0] * num_games

    replay = deque()
    replay_size = 0
    N = num_games
    opp_temps = np.zeros(N, dtype=np.float32)  # greedy move selection for the opponent
    wr_ema = 0.5
    last_wr_ref = 0.5
    last_ref_dec = 0  # decisive games in the most recent eval-vs-ref window

    for gen in range(gen0, gen0 + num_generations):
        gen_started = time.monotonic()
        next_generation.assign(gen + 1)
        opp_tag = _sample_pool(opp_net, pool_dir)  # this generation's adversary

        gen_pos = []
        completed_episodes = []
        search_metrics = []
        search_work = {}
        productive_attack = surge_attack = 0.0
        state_recs = []  # both players' state records for offline oracle relabeling
        game_lens, p1_wins = [], []  # p1_wins: one bool per DECISIVE game
        n_draw = 0
        learner_attack = learner_placements = 0
        # Learner b2b/combo economics, from p1's scorer around each committed placement.
        b2b_at_death, chain_runs = [], []
        n_deaths = 0

        for _t in range(horizon):
            temps_p1 = np.where(move_count < cfg.temp_moves, 1.0, 0.0).astype(
                np.float32
            )
            r1 = mcts.search([p[0] for p in pairs], 1.0, temps_p1)  # learner
            r2 = opp_mcts.search([p[1] for p in pairs], 1.0, opp_temps)  # pool opponent
            for search in (mcts, opp_mcts):
                for key, value in search.last_stats.items():
                    search_work[key] = search_work.get(key, 0) + value

            for g in range(N):
                a, b = r1[g], r2[g]
                e1, e2 = pairs[g]

                if a["dead"] or b["dead"]:
                    if a["dead"]:
                        b2b_at_death.append(e1._scorer._b2b)
                        n_deaths += 1
                    p1_died, p2_died = a["dead"], b["dead"]
                else:
                    pending[g]["p1"].append(_pos(a))
                    pending[g]["p2"].append(_pos(b))
                    if save_states_dir:
                        state_recs.append(_state_record(e1))
                        state_recs.append(_state_record(e2))
                    pre_b2b = e1._scorer._b2b
                    pre_b2b2 = e2._scorer._b2b
                    p1_died, p2_died, atk1, atk2 = _commit_and_exchange(
                        e1, e2, searcher, a["descriptor"], b["descriptor"], rng
                    )
                    prod1 = atk1 if e1._scorer._b2b == pre_b2b + 1 else 0.0
                    prod2 = atk2 if e2._scorer._b2b == pre_b2b2 + 1 else 0.0
                    for player, prod in (("p1", prod1), ("p2", prod2)):
                        pending[g][player][-1]["reward"] = cfg.w_attack * prod
                        pending[g][player][-1]["productive_attack"] = prod
                    productive_attack += prod1
                    surge_attack += (
                        pre_b2b if pre_b2b >= 4 and e1._scorer._b2b == -1 else 0
                    )
                    search_metrics.append(a)
                    post_b2b = e1._scorer._b2b
                    if post_b2b == pre_b2b + 1:
                        cur_chain[g] += 1
                    else:
                        if cur_chain[g] > 0:
                            chain_runs.append(cur_chain[g])
                        cur_chain[g] = 0
                    if p1_died:
                        b2b_at_death.append(pre_b2b)
                        n_deaths += 1
                    learner_attack += atk1
                    learner_placements += 1
                    move_count[g] += 1
                    cap = move_count[g] >= max_game_steps
                    if not (p1_died or p2_died or cap):
                        continue

                completed_episodes.append(
                    (
                        pending[g],
                        p1_died,
                        p2_died,
                        (state_observation(e1), state_observation(e2)),
                    )
                )
                if cur_chain[g] > 0:
                    chain_runs.append(cur_chain[g])
                cur_chain[g] = 0
                e1._reset()
                e2._reset()
                move_count[g] = 0
                pending[g] = {"p1": [], "p2": []}

        for rows, glen, p1_won, draw in _finalize_episodes(
            completed_episodes, net, mini_batch_size, n_step, cfg.gamma
        ):
            gen_pos.extend(rows)
            game_lens.append(glen)
            if draw:
                n_draw += 1
            else:
                p1_wins.append(p1_won)

        if save_states_dir and state_recs:
            os.makedirs(save_states_dir, exist_ok=True)
            shard = os.path.join(save_states_dir, f"shard_{gen}")
            save_states(state_recs, shard)
            print(f"Gen {gen}: saved {len(state_recs)} states to {shard}", flush=True)

        # Rate this gen's games BEFORE the training-update skip paths, so the
        # whole-history log never drops completed games (e.g. replay warmup).
        decisive = len(p1_wins)
        wins = int(sum(p1_wins))
        if whr is not None and opp_tag is not None and decisive + n_draw > 0:
            whr.record(gen, opp_tag, wins, decisive - wins, n_draw, ctx="pool")
        if gen % eval_interval == 0:
            ref_wins, ref_losses, ref_draws = _eval_vs_ref(
                eval_mcts,
                ref_mcts,
                eval_games,
                queue_size,
                max_len,
                max_game_steps,
                rng,
                searcher,
                9001 + gen * eval_games,
            )
            last_ref_dec = ref_wins + ref_losses
            if last_ref_dec:  # hold the previous value across all-draw evals
                last_wr_ref = ref_wins / last_ref_dec
            if whr is not None:
                whr.record(gen, "gen_0", ref_wins, ref_losses, ref_draws, ctx="eval")
        if whr is not None:
            write_gen = gen % 5 == 0
            whr.fit(gen=gen, full_sigma=write_gen)
            if write_gen:
                whr.to_json(ratings_path)

        search_stats = {
            f"search/{k}": search_work.get(k, 0) for k in ("seconds", "inference_calls")
        }
        search_stats["search/inference_batch_utilization"] = search_work.get(
            "inference_rows", 0
        ) / max(1, search_work.get("inference_capacity", 0))
        collection_stats = _collection_metrics(
            search_metrics,
            learner_attack,
            productive_attack,
            surge_attack,
            learner_placements,
            time.monotonic() - gen_started,
        )
        collection_stats.update(
            {
                **search_stats,
                "progress/optimization_seconds": 0.0,
                "progress/buffer_size": replay_size,
                "progress/completed_games": len(game_lens),
                "progress/updates": 0,
                "counts/n_deaths": n_deaths,
            }
        )
        n_new = len(gen_pos)
        if n_new == 0:
            log_step(OneVsOneCollectionLog(diagnostics=collection_stats), step=gen)
            print(f"Gen {gen}: no games completed; skipping update.", flush=True)
            continue

        boards = np.stack([p["board"] for p, *_ in gen_pos]).astype(np.float32)
        pieces = np.stack([p["pieces"] for p, *_ in gen_pos]).astype(np.int64)
        bcg = np.stack([p["bcg"] for p, *_ in gen_pos]).astype(np.float32)
        cand_pl = np.stack([p["cand_placements"] for p, *_ in gen_pos]).astype(
            np.float32
        )
        cand_mk = np.stack([p["cand_mask"] for p, *_ in gen_pos]).astype(bool)
        pi_tgt = np.stack([p["pi"] for p, *_ in gen_pos]).astype(np.float32)
        value_tgt = np.array([r[1] for r in gen_pos], dtype=np.float32)
        policy_mask = np.array([r[2] for r in gen_pos], dtype=np.float32)
        slots = np.array([p["slot"] for p, *_ in gen_pos], np.int32)
        hazard_target = np.stack([p["hazard_target"] for p, *_ in gen_pos])
        hazard_mask = np.stack([p["hazard_mask"] for p, *_ in gen_pos])
        # gen_pos interleaves both players; policy_mask==1 is the learner.
        lrn = policy_mask == 1.0
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
                "slot": slots,
                "hazard_target": hazard_target,
                "hazard_mask": hazard_mask,
            }
        )
        replay_size += n_new
        while replay_size > replay_capacity and len(replay) > 1:
            replay_size -= len(replay.popleft()["value_target"])

        collection_stats["progress/buffer_size"] = replay_size
        if replay_size < mini_batch_size:
            log_step(OneVsOneCollectionLog(diagnostics=collection_stats), step=gen)
            print(
                f"Gen {gen}: replay {replay_size} < batch {mini_batch_size}; skipping update.",
                flush=True,
            )
            continue

        optimization_started = time.monotonic()
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
        learner_idx = np.flatnonzero(lrn)[:mini_batch_size]
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

        # Average the optimization stats over the generation's minibatches.
        updates = 0
        acc = {}
        for batch in ds:
            step_out = train_step(
                net,
                batch,
                tf.constant(value_coef, tf.float32),
                tf.constant(risk_coef, tf.float32),
            )
            for k, v in step_out.items():
                acc.setdefault(k, []).append(float(v))
            updates += 1
        if not acc:
            print(f"Gen {gen}: no batch produced; skipping update.", flush=True)
            continue
        opt = {k: float(np.mean(v)) for k, v in acc.items()}

        if measure_kl:
            lp_after = _gen_log_probs(net, *gi).numpy()
            update_kl = float(
                (np.exp(lp_before) * (lp_before - lp_after)).sum(axis=-1).mean()
            )
        else:
            update_kl = 0.0
        optimization_seconds = time.monotonic() - optimization_started

        n_games = len(game_lens)
        win_rate = wins / decisive if decisive else 0.0
        app_learner = learner_attack / learner_placements if learner_placements else 0.0
        learner_rows = [p for p, _target, pm, *_ in gen_pos if pm]
        death_target = np.stack([p["death_target"] for p in learner_rows])
        death_observed = np.stack([p["death_observed"] for p in learner_rows])
        risk_stats = risk_calibration(
            np.stack([p["risk_prediction"] for p in learner_rows]),
            death_target,
            death_observed,
        )
        risk_search_stats = risk_calibration(
            np.stack([p["risk_curve"] for p in learner_rows]),
            death_target,
            death_observed,
        )
        extras = {
            **search_stats,
            "progress/optimization_seconds": optimization_seconds,
            **{f"risk/{k}": v for k, v in risk_stats.items()},
            **{
                f"risk_search/{k}": risk_search_stats[k]
                for k in ("predicted_h24", "brier_h24")
            },
            "risk/loss": opt["risk_loss"],
            **_collection_metrics(
                search_metrics,
                learner_attack,
                productive_attack,
                surge_attack,
                learner_placements,
                time.monotonic() - gen_started,
            ),
        }

        # Pool maintenance: EMA the decisive WR and grow the pool (gated). Rating
        # bookkeeping already ran pre-skip; a new snapshot registers + refits here
        # so ratings.json includes it immediately.
        if decisive > 0:
            wr_ema = 0.9 * wr_ema + 0.1 * win_rate
        if (
            gen > 0
            and gen % pool_interval == 0
            and decisive >= 8
            and wr_ema >= pool_wr_gate
        ):
            _save_pool(net, gen, pool_dir, max_pool_size, cfg, n_step)
            print(f"Saved opponent-pool gen_{gen} (wr_ema {wr_ema:.3f}).", flush=True)
            if whr is not None:
                whr.register_snapshot(f"gen_{gen}")
                whr.fit(gen=gen, full_sigma=True)
                whr.to_json(ratings_path)

        log_step(
            OneVsOneAZLog(
                policy_loss=opt["policy_loss"],
                value_loss=opt["value_loss"],
                entropy=opt["entropy"],
                update_kl=update_kl,
                explained_var=opt["explained_var"],
                grad_norm=opt["grad_norm"],
                win_rate_vs_ref=last_wr_ref,
                ref_decisive=last_ref_dec,
                avg_b2b=float(bcg[lrn, 0].mean()),
                b2b_at_death=_mean_or_none(b2b_at_death),
                chain_run_len=_mean_or_none(chain_runs),
                n_deaths=n_deaths,
                updates=updates,
                buffer_size=replay_size,
                completed_games=n_games,
                diagnostics=extras,
            ),
            step=gen,
        )
        print(
            f"Gen {gen} | Policy: {opt['policy_loss']:2.3f} | "
            f"Value: {opt['value_loss']:2.3f} | "
            f"Ent: {opt['entropy']:1.3f} | "
            f"WR(pool {opp_tag}): {win_rate:1.2f} | WRvsRef: {last_wr_ref:1.2f} | "
            f"Games: {n_games} | APP: {app_learner:1.3f} | "
            f"Productive APP: {extras['attack/productive_app']:1.3f} | Updates: {updates}",
            flush=True,
        )

        if gen % 5 == 0:
            save_checkpoint(manager, cfg, n_step)

    save_checkpoint(manager, cfg, n_step)
    finish(run)
