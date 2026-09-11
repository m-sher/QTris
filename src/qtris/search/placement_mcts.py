"""Batched C PUCT search with scalar or productive-attack/death-risk critics."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from qtris.data.placement_features import MCTS_CANDIDATE_CAPACITY
from qtris.search.cmcts import CMCTS, RISK_HORIZON


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    gamma: float = 0.99
    temp_moves: int = 12  # moves played at temperature 1 before switching to greedy
    w_attack: float = 0.006  # per-edge reward weight on credited attack
    w_death: float = (
        100.0  # terminal-edge penalty (raw attack units; same scale as a strong clear)
    )
    w_b2b: float = 0.0054  # b2b-build potential shaping; Phi=min(max(0,b2b),45), 0=off
    w_height: float = 0.06  # board potential on min(1, max_height/24), 0=off
    w_bumpiness: float = 0.03  # board potential on min(1, bumpiness/48), 0=off
    w_holes: float = 0.16  # board potential on min(1, holes/16), 0=off
    w_plain: float = 0.03  # cost of a non-difficult clear with nothing queued, 0=off
    q_norm: bool = True  # rank on per-tree min-max normalised Q
    fpu: float = 0.4  # unvisited child scores parent value minus this; <0 scores 0
    four_wide: bool = False  # hold the 4-wide side walls level on every in-tree board
    w_residual: float = (
        0.0  # four_wide: bonus per clearing edge into a residual-matched stack
    )
    leaves_per_round: int = (
        4  # intra-tree leaf batching: L leaves/tree/net-call (virtual loss)
    )
    risk_gate: bool = False
    risk_threshold: float = 0.10
    risk_margin: float = 0.05
    vloss: float = 1.0  # virtual-loss magnitude (scaled-Q units)


class PlacementMCTS:
    def __init__(self, net, cfg: MCTSConfig):
        self.net = net
        self.cfg = cfg

    def _net_eval(self, boards, pieces, bcg, pls, masks):
        # Pad to the fixed inference batch and discard padded outputs.
        nv = boards.shape[0]
        fb = self._fullb
        if nv < fb:
            p = fb - nv

            def z(a):
                return np.concatenate([a, np.zeros((p, *a.shape[1:]), a.dtype)])

            boards, pieces, bcg, pls, masks = (
                z(boards),
                z(pieces),
                z(bcg),
                z(pls),
                z(masks),
            )
        forward = (
            self.net.policy_attack_risk if self.cfg.risk_gate else self.net.policy_value
        )
        outputs = forward(
            (
                tf.constant(boards, tf.float32),
                tf.constant(pieces, tf.int64),
                tf.constant(bcg, tf.float32),
                tf.constant(pls, tf.float32),
                tf.constant(masks, tf.bool),
            )
        )
        logits, value = outputs[:2]
        risks = outputs[2].numpy()[:nv] if self.cfg.risk_gate else None
        return logits.numpy()[:nv], value.numpy()[:nv, 0], risks

    def _select_action(self, legal, counts, pi, temperature):
        c = counts[legal]
        if c.sum() <= 0:
            return int(legal[np.argmax(pi[legal])])
        if temperature <= 0.0:
            return int(legal[np.argmax(c)])
        probs = c ** (1.0 / temperature)
        probs = probs / probs.sum()
        return int(np.random.choice(legal, p=probs))

    def search(self, real_envs, return_scale, temperatures):
        """Search one move per game and return observations, visits, and gated actions.

        Counts retain all committed visits; pi and gate_mask describe final eligibility.
        Commit descriptor=(hold, rotation, column, landing_row, spin) with placement_step.
        """
        n = len(real_envs)
        self._fullb = n * min(
            16, max(1, self.cfg.leaves_per_round)
        )  # fixed net batch (see _net_eval)
        temps = np.broadcast_to(np.asarray(temperatures, dtype=np.float32), (n,))
        e0 = real_envs[0]
        engine = CMCTS(
            n,
            board_height=40,
            queue_size=e0._queue_size,
            max_holes=e0._max_holes,
            garbage_push_delay=e0._garbage_push_delay,
            # These flags govern who maintains the real state between moves, so the sim
            # sets both for itself: queued garbage lands on a non-clearing move in every
            # regime we play, and the sim extends its queue from the mirrored bag RNG.
            auto_push_garbage=1,
            auto_fill_queue=1,
            c_puct=self.cfg.c_puct,
            gamma=self.cfg.gamma,
            w_attack=self.cfg.w_attack,
            w_death=self.cfg.w_death,
            return_scale=float(return_scale),
            max_len=e0._max_len,
            num_simulations=self.cfg.num_simulations,
            leaves_per_round=self.cfg.leaves_per_round,
            vloss=self.cfg.vloss,
            w_b2b=self.cfg.w_b2b,
            q_norm=self.cfg.q_norm,
            w_height=self.cfg.w_height,
            w_bumpiness=self.cfg.w_bumpiness,
            fpu=self.cfg.fpu,
            w_holes=self.cfg.w_holes,
            w_plain=self.cfg.w_plain,
            four_wide=self.cfg.four_wide,
            w_residual=self.cfg.w_residual,
            risk_gate=self.cfg.risk_gate,
            risk_threshold=self.cfg.risk_threshold,
            risk_margin=self.cfg.risk_margin,
        )
        try:
            for i, env in enumerate(real_envs):
                engine.set_root(i, env)

            obs = [None] * n
            nv, req = engine.collect_roots()
            if nv:
                boards, pieces, bcg, pls, masks, tree_ids = req
                logits, values, risks = self._net_eval(boards, pieces, bcg, pls, masks)
                noise = np.zeros((nv, MCTS_CANDIDATE_CAPACITY), dtype=np.float32)
                for k in range(nv):
                    ls = np.flatnonzero(masks[k])
                    if ls.size:
                        noise[k, ls] = np.random.dirichlet(
                            [self.cfg.dirichlet_alpha] * ls.size
                        )
                engine.apply_roots(logits, values, noise, self.cfg.dirichlet_eps, risks)
                for k in range(nv):
                    obs[tree_ids[k]] = {
                        "board": boards[k].copy(),
                        "pieces": pieces[k].copy(),
                        "bcg": bcg[k].copy(),
                        "cand_placements": pls[k].copy(),
                        "cand_mask": masks[k].copy(),
                        "risk_prediction": risks[k].copy()
                        if risks is not None
                        else None,
                        "value": float(
                            values[k]
                        ),  # net root value, for the AZ return bootstrap
                    }

            stats = engine.progress()
            while np.any((stats[:, 5] > 0) & (stats[:, 0] < self.cfg.num_simulations)):
                before = stats[:, 0].copy()
                nv, req = engine.collect_leaves()
                if nv:
                    boards, pieces, bcg, pls, masks, _tree_ids = req
                    logits, values, risks = self._net_eval(
                        boards, pieces, bcg, pls, masks
                    )
                else:
                    logits = np.empty((0, MCTS_CANDIDATE_CAPACITY), np.float32)
                    values = np.empty(0, np.float32)
                    risks = np.empty(
                        (0, MCTS_CANDIDATE_CAPACITY, RISK_HORIZON), np.float32
                    )
                engine.apply_leaves(logits, values, risks)
                stats = engine.progress()
                if np.array_equal(before, stats[:, 0]):
                    raise RuntimeError(
                        "MCTS stalled before completing its simulation budget"
                    )

            pi, counts, desc, dead, root_value = engine.result()
            curves, eligible, breaks = engine.root_risks()
        finally:
            engine.destroy()

        results = []
        for i in range(n):
            if dead[i] or obs[i] is None:
                results.append({"dead": True})
                continue
            legal = np.flatnonzero((desc[i, :, 0] >= 0) & eligible[i])
            slot = self._select_action(legal, counts[i], pi[i], float(temps[i]))
            row = {
                "dead": False,
                "slot": slot,
                "gate_mask": eligible[i],
                "break_mask": breaks[i],
                "risk_curve": curves[i, slot],
                "risk_chosen": float(curves[i, slot, -1]),
                "risk_best_keep": float(
                    curves[i, obs[i]["cand_mask"] & ~breaks[i], -1].min()
                )
                if np.any(obs[i]["cand_mask"] & ~breaks[i])
                else None,
                "max_depth": int(stats[i, 1]),
                "completed_simulations": int(stats[i, 0]),
                "pi": pi[i],
                "counts": counts[i].copy(),
                "descriptor": tuple(int(x) for x in desc[i, slot]),
                "visits": int(counts[i].sum()),
                "v_search": float(root_value[i]),
                **obs[i],
            }
            results.append(row)
        return results

    def root_values(self, real_envs):
        """Net value of each env's current root state (no simulation), for the n-step return
        bootstrap at the collection horizon. Returns a (num_games,) array; 0 where the root has
        no legal move (dead). Costs one batched root eval - the first half of `search()`."""
        n = len(real_envs)
        self._fullb = n * min(16, max(1, self.cfg.leaves_per_round))
        e0 = real_envs[0]
        engine = CMCTS(
            n,
            board_height=40,
            queue_size=e0._queue_size,
            max_holes=e0._max_holes,
            garbage_push_delay=e0._garbage_push_delay,
            # These flags govern who maintains the real state between moves, so the sim
            # sets both for itself: queued garbage lands on a non-clearing move in every
            # regime we play, and the sim extends its queue from the mirrored bag RNG.
            auto_push_garbage=1,
            auto_fill_queue=1,
            c_puct=self.cfg.c_puct,
            gamma=self.cfg.gamma,
            w_attack=self.cfg.w_attack,
            w_death=self.cfg.w_death,
            return_scale=1.0,
            max_len=e0._max_len,
            num_simulations=self.cfg.num_simulations,
            leaves_per_round=self.cfg.leaves_per_round,
            vloss=self.cfg.vloss,
            w_b2b=self.cfg.w_b2b,
            q_norm=self.cfg.q_norm,
            w_height=self.cfg.w_height,
            w_bumpiness=self.cfg.w_bumpiness,
            fpu=self.cfg.fpu,
            w_holes=self.cfg.w_holes,
            w_plain=self.cfg.w_plain,
            four_wide=self.cfg.four_wide,
            w_residual=self.cfg.w_residual,
            risk_gate=self.cfg.risk_gate,
            risk_threshold=self.cfg.risk_threshold,
            risk_margin=self.cfg.risk_margin,
        )
        out = np.zeros(n, dtype=np.float32)
        try:
            for i, env in enumerate(real_envs):
                engine.set_root(i, env)
            nv, req = engine.collect_roots()
            if nv:
                boards, pieces, bcg, pls, masks, tree_ids = req
                _logits, values, _risks = self._net_eval(
                    boards, pieces, bcg, pls, masks
                )
                for k in range(nv):
                    out[tree_ids[k]] = values[k]
        finally:
            engine.destroy()
        return out
