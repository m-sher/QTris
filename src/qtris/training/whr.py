"""Whole-history rating (WHR) for the 1v1 opponent-pool AZ trainer.

A batch MAP refit over an append-only game log: a Bradley-Terry likelihood on the
Elo scale with gen_0 clamped at `init`, one learner rating per
generation-with-games chained by a Gaussian random-walk prior
(`drift` Elo per sqrt-gen), one constant rating per pool snapshot soft-tied to the
learner trajectory at its creation gen (`tie_sigma`), a weak level prior on the
first epoch (`level_sigma`, makes the fit well-posed with zero anchor games), and
a single context-offset parameter for exploration-temperature pool games vs
greedy eval games (`ctx_sigma` prior; identifiable because gen_0 plays in both).
Draws count as half a win. Sigmas are Laplace marginals from diag(inv(-H)).
"""

import glob
import json
import math
import os
import re

import numpy as np

SCALE = 400.0 / math.log(10.0)  # Elo per natural logistic unit

_GEN_RE = re.compile(r"^gen_(\d+)$")


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


class GameLog:
    """Append-only JSONL of rated game batches.

    Loading tolerates exactly one torn FINAL line (crash mid-append): it warns and
    truncates the file back to the last good newline. A bad line anywhere else is
    a hard error (real corruption, not a torn tail).
    """

    def __init__(self, path):
        self.path = path

    def load(self):
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return []
        lines = raw.split(b"\n")
        entries = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:  # torn tail: drop + truncate
                    print(
                        f"WHR: dropping torn trailing line in {self.path}", flush=True
                    )
                    good = raw[: raw.rfind(line)]
                    with open(self.path, "wb") as f:
                        f.write(good)
                    break
                raise
        return entries

    def append(self, entry):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class WHRBook:
    """Batch-refit rating book over a GameLog.

    ratings/sigmas/games are plain {id: value} dicts refreshed by fit(). fit()
    never raises into the trainer: on solver failure it keeps the last converged
    state and flags converged=False.
    """

    def __init__(
        self,
        log_path,
        pool_dir=None,
        anchor="gen_0",
        learner="learner",
        init=1500.0,
        drift=8.0,
        tie_sigma=70.0,
        level_sigma=350.0,
        ctx_sigma=50.0,
    ):
        self.log = GameLog(log_path)
        self.anchor = anchor
        self.learner = learner
        self.init = init
        self.drift = drift
        self.tie_sigma = tie_sigma
        self.level_sigma = level_sigma
        self.ctx_sigma = ctx_sigma

        self._entries = self.log.load()
        self._snapshots = set()  # registered snapshot ids (rated even with 0 games)
        if pool_dir:
            for p in glob.glob(os.path.join(pool_dir, "gen_*.index")):
                self.register_snapshot(os.path.basename(p)[: -len(".index")])
        for e in self._entries:
            for pid in (e["p1"], e["p2"]):
                self.register_snapshot(pid)

        self._warm = {}  # param name -> natural value, last CONVERGED fit only
        self.converged = True
        self.fitted_at_gen = None
        self.ctx_offset = 0.0  # fitted pool-context offset, Elo
        self.ratings = {learner: init, anchor: init}
        self.sigmas = {learner: level_sigma, anchor: 0.0}
        self.games = {learner: 0, anchor: 0}
        self._trajectory = []  # [[gen, rating, sigma], ...] from last full fit
        if self._entries:
            self.fit(full_sigma=True)

    # Bookkeeping

    @property
    def last_gen(self):
        """Highest gen in the log, or -1 (resume offset source)."""
        return max((e["gen"] for e in self._entries), default=-1)

    def register_snapshot(self, player_id):
        if player_id not in (self.learner, self.anchor):
            self._snapshots.add(player_id)

    def record(self, gen, opp_id, wins, losses, draws=0, ctx="pool"):
        """Append one learner-vs-opp batch; no-op on zero games."""
        if wins + losses + draws == 0:
            return
        self.register_snapshot(opp_id)
        entry = {
            "gen": int(gen),
            "p1": self.learner,
            "p2": opp_id,
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
            "ctx": ctx,
        }
        self.log.append(entry)
        self._entries.append(entry)

    # Fitting

    def _creation_gen(self, snap_id):
        m = _GEN_RE.match(snap_id)
        return int(m.group(1)) if m else 0

    def fit(self, gen=None, full_sigma=True):
        """MAP refit of the whole log. Updates ratings/sigmas/games in place."""
        if gen is not None:
            self.fitted_at_gen = int(gen)
        games = {self.learner: 0, self.anchor: 0}
        for s in self._snapshots:
            games.setdefault(s, 0)
        for e in self._entries:
            n = e["wins"] + e["losses"] + e["draws"]
            games[e["p1"]] = games.get(e["p1"], 0) + n
            games[e["p2"]] = games.get(e["p2"], 0) + n
        self.games = games

        epochs = sorted(
            {e["gen"] for e in self._entries if self.learner in (e["p1"], e["p2"])}
        )
        snaps = sorted(self._snapshots, key=self._creation_gen)
        if not epochs:  # nothing to fit: everyone at init, prior-width sigmas
            self.ratings = {self.learner: self.init, self.anchor: self.init}
            self.sigmas = {self.learner: self.level_sigma, self.anchor: 0.0}
            for s in snaps:
                self.ratings[s] = self.init
                self.sigmas[s] = self.tie_sigma
            self.converged = True
            self._trajectory = []
            return

        try:
            self._solve(epochs, snaps, full_sigma)
            self.converged = True
        except Exception as exc:  # never raise into the trainer
            print(f"WHR: fit failed ({exc}); keeping previous ratings.", flush=True)
            self.converged = False

    def _solve(self, epochs, snaps, full_sigma):
        # Layout: [snapshots..., ctx, epochs ascending]; latest epoch LAST so its
        # marginal variance is exactly 1/L[-1,-1]^2 from the Cholesky factor.
        S, E = len(snaps), len(epochs)
        n = S + 1 + E
        snap_idx = {s: i for i, s in enumerate(snaps)}
        ctx_i = S
        ep_idx = {g: S + 1 + i for i, g in enumerate(epochs)}

        # Snapshot tie target: first epoch AFTER creation gen (weights are saved
        # post-update, so gen_k plays like the learner at k+1); fall back to the
        # last epoch <= k.
        def tie_epoch(k):
            after = [g for g in epochs if g > k]
            if after:
                return after[0]
            before = [g for g in epochs if g <= k]
            return before[-1] if before else None

        # Entry -> (index list, sign list, s, n) with anchor as fixed 0 (absent).
        def resolve(pid, g):
            if pid == self.learner:
                return ep_idx[g]
            if pid == self.anchor:
                return None
            return snap_idx[pid]

        terms = []
        for e in self._entries:
            i1, i2 = resolve(e["p1"], e["gen"]), resolve(e["p2"], e["gen"])
            idx, sgn = [], []
            if i1 is not None:
                idx.append(i1)
                sgn.append(1.0)
            if i2 is not None:
                idx.append(i2)
                sgn.append(-1.0)
            if e.get("ctx", "pool") == "pool" and e["p1"] == self.learner:
                idx.append(ctx_i)
                sgn.append(1.0)
            s = e["wins"] + 0.5 * e["draws"]
            terms.append((idx, sgn, s, s + e["losses"] + 0.5 * e["draws"]))

        # Prior precisions (natural units).
        drift_prec = [
            1.0 / ((self.drift / SCALE) ** 2 * (epochs[i + 1] - epochs[i]))
            for i in range(E - 1)
        ]
        level_prec = (SCALE / self.level_sigma) ** 2
        ctx_prec = (SCALE / self.ctx_sigma) ** 2
        # Tie variance grows with the random-walk distance between the snapshot's
        # creation and its nearest epoch.
        ties = []  # (snap param, epoch param, precision)
        for s in snaps:
            k = self._creation_gen(s)
            g = tie_epoch(k)
            var = self.tie_sigma**2
            if g is not None:
                var += self.drift**2 * abs(g - (k + 1))
            ties.append(
                (snap_idx[s], ep_idx[g] if g is not None else None, SCALE**2 / var)
            )

        def neg_logpost(x):
            ll = 0.0
            for idx, sgn, s, tot in terms:
                d = sum(x[i] * g for i, g in zip(idx, sgn))
                ll += s * _log_sigmoid(d) + (tot - s) * _log_sigmoid(-d)
            lp = -0.5 * level_prec * x[S + 1] ** 2
            for i, p in enumerate(drift_prec):
                lp += -0.5 * p * (x[S + 2 + i] - x[S + 1 + i]) ** 2
            for si, ei, tp in ties:
                diff = x[si] - (x[ei] if ei is not None else 0.0)
                lp += -0.5 * tp * diff**2
            lp += -0.5 * ctx_prec * x[ctx_i] ** 2
            return -(ll + lp)

        def grad_hess(x):
            g = np.zeros(n)
            A = np.zeros((n, n))  # A = -Hessian of log-posterior (PD)
            for idx, sgn, s, tot in terms:
                d = sum(x[i] * sg for i, sg in zip(idx, sgn))
                p = 1.0 / (1.0 + np.exp(-d))
                gd = s - tot * p
                hd = tot * p * (1.0 - p)
                for i, sg in zip(idx, sgn):
                    g[i] += sg * gd
                    for j, sg2 in zip(idx, sgn):
                        A[i, j] += hd * sg * sg2
            i0 = S + 1
            g[i0] -= level_prec * x[i0]
            A[i0, i0] += level_prec
            for i, p in enumerate(drift_prec):
                a, b = S + 1 + i, S + 2 + i
                diff = x[b] - x[a]
                g[a] += p * diff
                g[b] -= p * diff
                A[a, a] += p
                A[b, b] += p
                A[a, b] -= p
                A[b, a] -= p
            for si, ei, tp in ties:
                diff = x[si] - (x[ei] if ei is not None else 0.0)
                g[si] -= tp * diff
                A[si, si] += tp
                if ei is not None:
                    g[ei] += tp * diff
                    A[ei, ei] += tp
                    A[si, ei] -= tp
                    A[ei, si] -= tp
            g[ctx_i] -= ctx_prec * x[ctx_i]
            A[ctx_i, ctx_i] += ctx_prec
            return g, A

        # Warm start: prior fit values; new epochs at their predecessor, new
        # snapshots at their tie target.
        x = np.zeros(n)
        for i, s in enumerate(snaps):
            x[i] = self._warm.get(f"snap:{s}", np.nan)
        x[ctx_i] = self._warm.get("ctx", 0.0)
        prev = 0.0
        for g_, i in ep_idx.items():
            v = self._warm.get(f"ep:{g_}", np.nan)
            if np.isnan(v):
                v = prev
            x[i] = prev = v
        for si, ei, _tp in ties:
            if np.isnan(x[si]):
                x[si] = x[ei] if ei is not None else 0.0

        # Damped Newton with jitter-retry Cholesky.
        obj = neg_logpost(x)
        L = None
        for _ in range(50):
            g, A = grad_hess(x)
            L = self._chol_jitter(A)
            step = self._chol_solve(L, g)
            t = 1.0
            for _bt in range(10):
                x_new = x + t * step
                obj_new = neg_logpost(x_new)
                if obj_new <= obj + 1e-12:
                    break
                t *= 0.5
            x, obj = x_new, obj_new
            if np.max(np.abs(t * step)) < 1e-6:
                break

        _, A = grad_hess(x)
        L = self._chol_jitter(A)

        ratings = {self.learner: self.init + SCALE * x[S + E], self.anchor: self.init}
        for s in snaps:
            ratings[s] = self.init + SCALE * x[snap_idx[s]]

        sigmas = {self.anchor: 0.0}
        # Latest-epoch marginal is exact and O(1) from the last Cholesky pivot.
        sigmas[self.learner] = SCALE / L[-1, -1]
        trajectory = self._trajectory
        if full_sigma:
            var = np.diag(np.linalg.inv(A))
            sigmas[self.learner] = SCALE * math.sqrt(var[S + E])
            for s in snaps:
                sigmas[s] = SCALE * math.sqrt(var[snap_idx[s]])
            trajectory = [
                [g_, self.init + SCALE * x[i], SCALE * math.sqrt(var[i])]
                for g_, i in ep_idx.items()
            ]
        else:
            for s in snaps:
                sigmas[s] = self.sigmas.get(s, self.tie_sigma)

        # Commit state only after every solve/inverse step has succeeded, so a
        # failure leaves the previous fit fully intact (never a mixed state).
        self._warm = {f"snap:{s}": x[snap_idx[s]] for s in snaps}
        self._warm["ctx"] = x[ctx_i]
        self._warm.update({f"ep:{g_}": x[i] for g_, i in ep_idx.items()})
        self.ctx_offset = SCALE * x[ctx_i]
        self.ratings = ratings
        self.sigmas = sigmas
        self._trajectory = trajectory

    @staticmethod
    def _chol_jitter(A):
        scale = float(np.mean(np.diag(A))) or 1.0
        for jit in (0.0, 1e-8, 1e-6, 1e-4):
            try:
                return np.linalg.cholesky(A + jit * scale * np.eye(len(A)))
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError("Hessian not PD after jitter escalation")

    @staticmethod
    def _chol_solve(L, b):
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)

    # Reporting

    def present_summary(self, present_ids):
        """Derived log metrics for the present pool members."""
        r_l = self.ratings[self.learner]
        pool = [
            self.ratings[g]
            for g in present_ids
            if g in self.ratings and g not in (self.learner, self.anchor)
        ]
        best_pool = max(pool) if pool else r_l
        return {
            "best_pool": best_pool,
            "learner_minus_ref": r_l - self.ratings[self.anchor],
            "gap_to_pool": r_l - best_pool,
        }

    def to_json(self, path):
        """Atomically write ratings.json (tmp + os.replace)."""
        payload = {
            "anchor": self.anchor,
            "learner": self.learner,
            "init": self.init,
            "drift": self.drift,
            "tie_sigma": self.tie_sigma,
            "fitted_at_gen": self.fitted_at_gen,
            "converged": self.converged,
            "ctx_offset": round(self.ctx_offset, 2),
            "ratings": {
                pid: {
                    "rating": round(self.ratings[pid], 2),
                    "sigma": round(self.sigmas.get(pid, self.tie_sigma), 2),
                    "games": self.games.get(pid, 0),
                }
                for pid in self.ratings
            },
            "trajectory": [
                [g, round(r, 1), round(s, 1)] for g, r, s in self._trajectory
            ],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
