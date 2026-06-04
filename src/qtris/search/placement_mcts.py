"""PUCT MCTS over candidate placements for the AlphaZero placement pipeline.

The net's policy head supplies edge priors and its value head evaluates leaves;
expansion clones the real env and steps a candidate sequence so child states match
the training distribution exactly (reusing `clone_sim_env`/`net_input_from_env` from
`placement_search`). Candidates come from the env's own pathfinder (the full legal
placement set, no heuristic death-pruning), so MCTS+net own all evaluation - closest
to canonical AlphaZero. Search runs across N self-play games at once so every
simulation's leaf evaluations batch into a single net call.

Reward-bearing backup: each edge stores the (scaled) immediate reward observed on the
step, so the backed-up return is `G = r + gamma * G` (MuZero-style) rather than a pure
terminal outcome. Q values are min-max normalized per tree before entering PUCT so the
exploration term stays calibrated against an unknown reward scale.
"""

from dataclasses import dataclass

import numpy as np

from TetrisEnv.Moves import Keys
from qtris.data.placement_features import CANDIDATE_CAPACITY, build_placement_inference
from qtris.search.placement_search import (
    _policy_value_batch,
    clone_sim_env,
    net_input_from_env,
)

HARD_DROP = Keys.HARD_DROP
REWARD_CLIP = 10.0
ROW_NORM = 23  # board height - 1


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    gamma: float = 0.99
    temp_moves: int = 12  # moves played at temperature 1 before switching to greedy


def _enumerate(env):
    """Full legal placement set via the env's pathfinder (no death-pruning; MCTS
    discovers death itself). Returns (placements[128,18], mask[128], seqs[128,max_len])
    or None if the state has no legal placement (dead)."""
    scores, rows, seqs = env._enumerate_placement_candidates()
    valid = np.flatnonzero(scores > -1e29)
    if valid.size == 0:
        return None
    queue = env._queue
    return build_placement_inference(
        valid,
        scores[valid],
        rows[valid],
        seqs[valid],
        active_piece=env._active_piece.piece_type.value,
        hold_piece=env._hold_piece.value,
        queue0=int(queue[0].value) if queue else 0,
        row_norm=ROW_NORM,
        max_len=env._max_len,
    )


class _MinMaxStats:
    """Running min/max of Q across one tree, to normalize Q into [0,1] for PUCT."""

    def __init__(self):
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class _Node:
    __slots__ = (
        "env",
        "terminal",
        "value",
        "seqs",
        "legal",
        "prior",
        "N",
        "W",
        "Q",
        "edge_reward",
        "children",
        "stats",
    )

    def __init__(self, env, terminal, stats):
        self.env = env
        self.terminal = terminal
        self.stats = stats
        self.value = 0.0
        self.seqs = None
        self.legal = np.empty(0, dtype=np.int64)
        self.children = {}
        self.prior = np.zeros(CANDIDATE_CAPACITY, dtype=np.float32)
        self.N = np.zeros(CANDIDATE_CAPACITY, dtype=np.float32)
        self.W = np.zeros(CANDIDATE_CAPACITY, dtype=np.float32)
        self.Q = np.zeros(CANDIDATE_CAPACITY, dtype=np.float32)
        self.edge_reward = np.zeros(CANDIDATE_CAPACITY, dtype=np.float32)


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _legal_slots(mask, seqs):
    """Legal candidate slots: enumerated AND with a hard-drop in their key sequence."""
    legal = np.flatnonzero(mask)
    return np.array([s for s in legal if np.any(seqs[s] == HARD_DROP)], dtype=np.int64)


class PlacementMCTS:
    def __init__(self, net, cfg: MCTSConfig):
        self.net = net
        self.cfg = cfg

    def _scale_reward(self, reward, return_scale):
        return float(np.clip(reward / (return_scale + 1e-8), -REWARD_CLIP, REWARD_CLIP))

    def _set_node(self, node, value, logits, enum):
        """Populate a freshly enumerated, non-terminal node with value + priors."""
        placements, mask, seqs = enum
        legal = _legal_slots(mask, seqs)
        if legal.size == 0:
            node.terminal = True
            node.env = None
            return False
        node.value = float(value)
        node.seqs = seqs
        node.legal = legal
        node.prior[legal] = _softmax(logits[legal])
        return True

    def _select(self, node):
        legal = node.legal
        n, q, p = node.N[legal], node.Q[legal], node.prior[legal]
        total = float(n.sum())
        q_norm = np.where(n > 0, node.stats.normalize(q), 0.0)
        u = self.cfg.c_puct * p * np.sqrt(total + 1e-8) / (1.0 + n)
        return int(legal[np.argmax(q_norm + u)])

    def _descend(self, root):
        path = []
        node = root
        while True:
            if node.terminal or node.legal.size == 0:
                return path, None, None, "terminal"
            slot = self._select(node)
            path.append((node, slot))
            child = node.children.get(slot)
            if child is None:
                return path, node, slot, "expand"
            node = child

    def _backup(self, path, leaf_value):
        g = leaf_value
        for node, slot in reversed(path):
            g = node.edge_reward[slot] + self.cfg.gamma * g
            node.N[slot] += 1.0
            node.W[slot] += g
            node.Q[slot] = node.W[slot] / node.N[slot]
            node.stats.update(node.Q[slot])

    def _simulate_batch(self, roots, return_scale):
        """One simulation per live game: descend, then batch-expand all leaves."""
        requests = []  # (path, parent, slot)
        for root in roots:
            if root is None:
                continue
            path, parent, slot, kind = self._descend(root)
            if kind == "terminal":
                self._backup(path, 0.0)
            else:
                requests.append((path, parent, slot))
        if not requests:
            return

        stepped = []
        for path, parent, slot in requests:
            child_env = clone_sim_env(parent.env)
            ts = child_env._step(np.asarray(parent.seqs[slot], dtype=np.int64))
            r = self._scale_reward(float(ts.reward["total_reward"]), return_scale)
            stepped.append(
                {
                    "path": path,
                    "parent": parent,
                    "slot": slot,
                    "env": child_env,
                    "reward": r,
                    "terminal": bool(ts.is_last()),
                    "enum": None,
                }
            )

        to_eval = []
        for k, rec in enumerate(stepped):
            if rec["terminal"]:
                continue
            enum = _enumerate(rec["env"])
            if enum is None:
                rec["terminal"] = True
            else:
                rec["enum"] = enum
                to_eval.append(k)

        logits = values = None
        if to_eval:
            boards, pieces, bcgs, pls, msks = [], [], [], [], []
            for k in to_eval:
                b, p, g = net_input_from_env(stepped[k]["env"])
                boards.append(b)
                pieces.append(p)
                bcgs.append(g)
                pls.append(stepped[k]["enum"][0])
                msks.append(stepped[k]["enum"][1])
            logits, values = _policy_value_batch(
                self.net, boards, pieces, bcgs, pls, msks
            )
        eval_pos = {k: pos for pos, k in enumerate(to_eval)}

        for k, rec in enumerate(stepped):
            parent, slot = rec["parent"], rec["slot"]
            parent.edge_reward[slot] = rec["reward"]
            if rec["terminal"]:
                parent.children[slot] = _Node(None, True, parent.stats)
                self._backup(rec["path"], 0.0)
                continue
            pos = eval_pos[k]
            child = _Node(rec["env"], False, parent.stats)
            if self._set_node(child, values[pos], logits[pos], rec["enum"]):
                parent.children[slot] = child
                self._backup(rec["path"], child.value)
            else:
                parent.children[slot] = child  # dead (no legal): terminal, value 0
                self._backup(rec["path"], 0.0)

    def _select_action(self, legal, counts, prior, temperature):
        if counts.sum() <= 0:
            return int(legal[np.argmax(prior[legal])])
        if temperature <= 0.0:
            return int(legal[np.argmax(counts)])
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        return int(np.random.choice(legal, p=probs))

    def search(self, real_envs, return_scale, temperatures):
        """Run MCTS for one move across all games. `temperatures` is a per-game play
        temperature (scalar broadcasts). Returns one result dict per game: either
        {dead: True} or {dead: False, pi, slot, key_sequence, visits, board, pieces,
        bcg, cand_placements, cand_mask}."""
        n = len(real_envs)
        temps = np.broadcast_to(np.asarray(temperatures, dtype=np.float32), (n,))
        clones = [clone_sim_env(e) for e in real_envs]
        enums = [_enumerate(c) for c in clones]

        valid = [i for i, e in enumerate(enums) if e is not None]
        logits = values = None
        if valid:
            boards, pieces, bcgs, pls, msks = [], [], [], [], []
            for i in valid:
                b, p, g = net_input_from_env(clones[i])
                boards.append(b)
                pieces.append(p)
                bcgs.append(g)
                pls.append(enums[i][0])
                msks.append(enums[i][1])
            logits, values = _policy_value_batch(
                self.net, boards, pieces, bcgs, pls, msks
            )

        roots = [None] * n
        obs = [None] * n
        for pos, i in enumerate(valid):
            node = _Node(clones[i], False, _MinMaxStats())
            if not self._set_node(node, values[pos], logits[pos], enums[i]):
                continue  # dead root (no legal placement)
            legal = node.legal
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * legal.size)
            node.prior[legal] = (1.0 - self.cfg.dirichlet_eps) * node.prior[
                legal
            ] + self.cfg.dirichlet_eps * noise
            roots[i] = node
            placements, mask, _ = enums[i]
            b, p, g = net_input_from_env(clones[i])
            obs[i] = {
                "board": b[0],
                "pieces": p[0],
                "bcg": g[0],
                "cand_placements": placements,
                "cand_mask": mask,
            }

        for _ in range(self.cfg.num_simulations):
            self._simulate_batch(roots, return_scale)

        results = []
        for i in range(n):
            node = roots[i]
            if node is None:
                results.append({"dead": True})
                continue
            legal = node.legal
            counts = node.N[legal]
            pi = np.zeros(CANDIDATE_CAPACITY, dtype=np.float32)
            pi[legal] = counts / counts.sum() if counts.sum() > 0 else node.prior[legal]
            slot = self._select_action(legal, counts, node.prior, float(temps[i]))
            results.append(
                {
                    "dead": False,
                    "pi": pi,
                    "slot": slot,
                    "key_sequence": node.seqs[slot].astype(np.int64),
                    "visits": int(counts.sum()),
                    **obs[i],
                }
            )
        return results
