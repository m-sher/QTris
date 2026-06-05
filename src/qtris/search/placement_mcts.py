"""PUCT MCTS over candidate placements, driven by the fully-C engine in `b2b_search.c`.

The whole simulation loop (descend / step / enumerate / backup) runs in C on a compact
bitboard+scalars node, OpenMP-threaded across the N self-play games; only the TF policy/value
net stays in Python. Per move: build one C tree per game, evaluate the roots in one batched net
call (+ Dirichlet noise), then for each simulation round `collect_leaves` -> one net call ->
`apply_leaves` until the budget is spent, and read out per-root visit counts.

Reward is attack + b2b only: per-edge `w_attack * attack` (surge + combo already fold into
`compute_attack`'s attack), leaf bootstrap `v + w_b2b * max(0, b2b_leaf)` (unrealized-hoard
credit). Q values are min-max normalized per tree so PUCT's exploration term stays calibrated.
Dirichlet root noise and final action sampling are generated here in Python and passed into C.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from qtris.data.placement_features import CANDIDATE_CAPACITY
from qtris.search.cmcts import CMCTS


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    gamma: float = 0.99
    temp_moves: int = 12  # moves played at temperature 1 before switching to greedy
    w_attack: float = 1.0  # per-edge reward weight on attack
    w_b2b: float = 1.0  # leaf-bootstrap weight on max(0, b2b)
    w_death: float = (
        5.0  # terminal-edge penalty (raw attack units; same scale as a strong clear)
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
        # shape: forward is ~flat in batch on GPU (~1.4ms at 16..256), but each *new* batch size
        # triggers a ~2s XLA recompile. Without this, the per-round leaf count varies and the
        # recompiles swamp the call-count savings. Padded rows are masked off and sliced away.
        nv = boards.shape[0]
        fb = self._fullb
        if nv < fb:
            p = fb - nv
            z = lambda a: np.concatenate([a, np.zeros((p, *a.shape[1:]), a.dtype)])  # noqa: E731
            boards, pieces, bcg, pls, masks = (
                z(boards),
                z(pieces),
                z(bcg),
                z(pls),
                z(masks),
            )
        logits, value = self.net.policy_value(
            (
                tf.constant(boards, tf.float32),
                tf.constant(pieces, tf.int64),
                tf.constant(bcg, tf.float32),
                tf.constant(pls, tf.float32),
                tf.constant(masks, tf.bool),
            )
        )
        return logits.numpy()[:nv], value.numpy()[:nv, 0]

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
        {dead: True} or {dead: False, pi, slot, descriptor, visits, board, pieces, bcg,
        cand_placements, cand_mask}. `descriptor` = (is_hold, rot, norm_col, landing_row,
        spin); commit the real move via `placement_step(env, searcher, descriptor)`."""
        n = len(real_envs)
        self._fullb = n * max(
            1, self.cfg.leaves_per_round
        )  # fixed net batch (see _net_eval)
        temps = np.broadcast_to(np.asarray(temperatures, dtype=np.float32), (n,))
        e0 = real_envs[0]
        engine = CMCTS(
            n,
            board_height=24,
            queue_size=e0._queue_size,
            max_height=e0._max_height,
            max_holes=e0._max_holes,
            garbage_push_delay=e0._garbage_push_delay,
            auto_push_garbage=int(e0._auto_push_garbage),
            auto_fill_queue=int(e0._auto_fill_queue),
            c_puct=self.cfg.c_puct,
            gamma=self.cfg.gamma,
            w_attack=self.cfg.w_attack,
            w_b2b=self.cfg.w_b2b,
            w_death=self.cfg.w_death,
            return_scale=float(return_scale),
            max_len=e0._max_len,
            num_simulations=self.cfg.num_simulations,
            leaves_per_round=self.cfg.leaves_per_round,
            vloss=self.cfg.vloss,
        )
        try:
            for i, env in enumerate(real_envs):
                engine.set_root(i, env)

            obs = [None] * n
            nv, req = engine.collect_roots()
            if nv:
                boards, pieces, bcg, pls, masks, tree_ids = req
                logits, values = self._net_eval(boards, pieces, bcg, pls, masks)
                noise = np.zeros((nv, CANDIDATE_CAPACITY), dtype=np.float32)
                for k in range(nv):
                    ls = np.flatnonzero(masks[k])
                    if ls.size:
                        noise[k, ls] = np.random.dirichlet(
                            [self.cfg.dirichlet_alpha] * ls.size
                        )
                engine.apply_roots(logits, values, noise, self.cfg.dirichlet_eps)
                for k in range(nv):
                    obs[tree_ids[k]] = {
                        "board": boards[k].copy(),
                        "pieces": pieces[k].copy(),
                        "bcg": bcg[k].copy(),
                        "cand_placements": pls[k].copy(),
                        "cand_mask": masks[k].copy(),
                    }

            lpr = max(1, self.cfg.leaves_per_round)
            rounds = (self.cfg.num_simulations + lpr - 1) // lpr  # ceil: L leaves/round
            for _ in range(rounds):
                nv, req = engine.collect_leaves()
                if nv == 0:
                    break
                boards, pieces, bcg, pls, masks, tree_ids = req
                logits, values = self._net_eval(boards, pieces, bcg, pls, masks)
                engine.apply_leaves(logits, values)

            pi, counts, desc, dead = engine.result()
        finally:
            engine.destroy()

        results = []
        for i in range(n):
            if dead[i] or obs[i] is None:
                results.append({"dead": True})
                continue
            legal = np.flatnonzero(desc[i, :, 0] >= 0)
            slot = self._select_action(legal, counts[i], pi[i], float(temps[i]))
            results.append(
                {
                    "dead": False,
                    "pi": pi[i],
                    "slot": slot,
                    "descriptor": tuple(int(x) for x in desc[i, slot]),
                    "visits": int(counts[i].sum()),
                    **obs[i],
                }
            )
        return results
