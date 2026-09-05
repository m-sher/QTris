"""PUCT MCTS over candidate placements, driven by the fully-C engine in `b2b_search.c`.

The whole simulation loop (descend / step / enumerate / backup) runs in C on a compact
bitboard+scalars node, OpenMP-threaded across the N self-play games; only the TF policy/value
net stays in Python. Per move: build one C tree per game, evaluate the roots in one batched net
call (+ Dirichlet noise), then for each simulation round `collect_leaves` -> one net call ->
`apply_leaves` until the budget is spent, and read out per-root visit counts plus the
shaping-free root value (leaf values + death edges only, in the same return_scale units
as Q).

Reward is per-edge `w_attack * credit`, where credit is a difficult clear's whole attack
and only the rows a non-difficult clear cancels from the own garbage queue (combo and the
b2b-break surge are already inside `compute_attack`'s attack), minus `w_plain` for a
non-difficult clear made with nothing queued, plus two potential differences:
`w_b2b * (gamma*Phi(child) - Phi(parent))` with `Phi = min(max(0, b2b), 45)`, and
`pen(parent) - gamma*pen(child)` with `pen = w_height * min(1, max_height/24) +
w_bumpiness * min(1, bumpiness/48) + w_holes * min(1, holes/16)`; terminal edges add
`-w_death` and read both potentials as 0. In four_wide mode a clearing edge that leaves
the middle stack matching a residual template adds `w_residual`. The shaped channel
bootstraps on the net value plus `w_attack * attack_window * attack_app_cap /
return_scale` times the attack head; the shaping-free channel on the net value alone.
PUCT ranks on per-tree min-max normalised Q when `q_norm`, raw return_scale units
otherwise; an unvisited child scores its parent's net value minus
`fpu`, floored at the tree minimum under `q_norm`. Dirichlet noise + sampling stay in
Python.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from qtris.data.placement_features import MCTS_CANDIDATE_CAPACITY
from qtris.search.cmcts import CMCTS


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    gamma: float = 0.99
    temp_moves: int = 12  # moves played at temperature 1 before switching to greedy
    w_attack: float = 0.006  # per-edge reward weight on credited attack
    attack_window: int = 14  # placements the attack head spans; 0 = head off in search
    attack_app_cap: float = (
        2.0  # attack per placement at which the head's target reads 1
    )
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
    vloss: float = 1.0  # virtual-loss magnitude (scaled-Q units)


class PlacementMCTS:
    def __init__(self, net, cfg: MCTSConfig):
        self.net = net
        self.cfg = cfg

    def _net_eval(self, boards, pieces, bcg, pls, masks):
        # Pad to a fixed batch (num_trees * leaves_per_round) so the jit_compiled net sees one
        # shape: each new batch size triggers an XLA recompile. Padded rows are masked off
        # and sliced away.
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
        logits, value, attack = self.net.policy_value(
            (
                tf.constant(boards, tf.float32),
                tf.constant(pieces, tf.int64),
                tf.constant(bcg, tf.float32),
                tf.constant(pls, tf.float32),
                tf.constant(masks, tf.bool),
            )
        )
        attack = np.clip(attack.numpy()[:nv, 0], 0.0, 1.0)
        return logits.numpy()[:nv], value.numpy()[:nv, 0], attack

    def _attack_coef(self, return_scale):
        """Weight of the attack head in a node's selection value, in scaled-Q units."""
        if self.cfg.attack_window <= 0:
            return 0.0
        return (
            self.cfg.w_attack
            * self.cfg.attack_window
            * self.cfg.attack_app_cap
            / (float(return_scale) + 1e-8)
        )

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
        """Run MCTS for one move across all games. `temperatures` is a per-game play
        temperature (scalar broadcasts). Returns one result dict per game: either
        {dead: True} or {dead: False, pi, counts, descriptor, visits, value, a_root,
        v_search, board, pieces, bcg, cand_placements, cand_mask}. `descriptor` =
        (is_hold, rot, norm_col, landing_row, spin); commit the real move via
        `placement_step(env, searcher, descriptor)`. `counts` carries the root visit
        counts alongside the normalized `pi`; `v_search` is the post-search shaping-free
        root value and `a_root` the net's attack head at the root."""
        n = len(real_envs)
        self._fullb = n * max(
            1, self.cfg.leaves_per_round
        )  # fixed net batch (see _net_eval)
        temps = np.broadcast_to(np.asarray(temperatures, dtype=np.float32), (n,))
        a_coef = self._attack_coef(return_scale)
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
        )
        try:
            for i, env in enumerate(real_envs):
                engine.set_root(i, env)

            obs = [None] * n
            nv, req = engine.collect_roots()
            if nv:
                boards, pieces, bcg, pls, masks, tree_ids = req
                logits, values, attacks = self._net_eval(
                    boards, pieces, bcg, pls, masks
                )
                noise = np.zeros((nv, MCTS_CANDIDATE_CAPACITY), dtype=np.float32)
                for k in range(nv):
                    ls = np.flatnonzero(masks[k])
                    if ls.size:
                        noise[k, ls] = np.random.dirichlet(
                            [self.cfg.dirichlet_alpha] * ls.size
                        )
                engine.apply_roots(
                    logits, values + a_coef * attacks, noise, self.cfg.dirichlet_eps
                )
                for k in range(nv):
                    obs[tree_ids[k]] = {
                        "board": boards[k].copy(),
                        "pieces": pieces[k].copy(),
                        "bcg": bcg[k].copy(),
                        "cand_placements": pls[k].copy(),
                        "cand_mask": masks[k].copy(),
                        "value": float(
                            values[k]
                        ),  # net root value, for the AZ return bootstrap
                        "a_root": float(attacks[k]),
                    }

            lpr = max(1, self.cfg.leaves_per_round)
            rounds = (self.cfg.num_simulations + lpr - 1) // lpr  # ceil: L leaves/round
            for _ in range(rounds):
                nv, req = engine.collect_leaves()
                if nv == 0:
                    break
                boards, pieces, bcg, pls, masks, tree_ids = req
                logits, values, attacks = self._net_eval(
                    boards, pieces, bcg, pls, masks
                )
                engine.apply_leaves(logits, values + a_coef * attacks, values)

            pi, counts, desc, dead, root_value = engine.result()
        finally:
            engine.destroy()

        results = []
        for i in range(n):
            if dead[i] or obs[i] is None:
                results.append({"dead": True})
                continue
            legal = np.flatnonzero(desc[i, :, 0] >= 0)
            slot = self._select_action(legal, counts[i], pi[i], float(temps[i]))
            row = {
                "dead": False,
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
        self._fullb = n * max(1, self.cfg.leaves_per_round)
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
        )
        out = np.zeros(n, dtype=np.float32)
        try:
            for i, env in enumerate(real_envs):
                engine.set_root(i, env)
            nv, req = engine.collect_roots()
            if nv:
                boards, pieces, bcg, pls, masks, tree_ids = req
                _logits, values, _attacks = self._net_eval(
                    boards, pieces, bcg, pls, masks
                )
                for k in range(nv):
                    out[tree_ids[k]] = values[k]
        finally:
            engine.destroy()
        return out
