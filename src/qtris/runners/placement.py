"""Rollout runner for placement-model RL (single-player).

The action is a candidate placement slot and the value comes from the merged
`PlacementPolicyValueNet`. The per-step candidate set (dense by action index) is
produced inside each env (`placement_candidates=True`), so the C search parallelizes
across the rollout subprocesses; the cheap 18-dim encoding is done here via
`build_placement_inference`.
"""

from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.placement_features import build_placement_inference

ROW_NORM = 23  # board height - 1


class PlacementRunner:
    def __init__(
        self,
        queue_size: int,
        max_holes: Optional[int],
        max_height: int,
        max_steps: int,
        max_len: int,
        num_steps: int,
        num_envs: int,
        garbage_chance_min: float,
        garbage_chance_max: float,
        garbage_rows_min: int,
        garbage_rows_max: int,
        net: Any,
        temperature: float = 1.0,
        seed: int = 123,
        num_row_tiers: int = 2,
    ) -> None:
        self._queue_size = queue_size
        self._max_len = max_len
        self._num_steps = num_steps
        self._num_envs = num_envs
        self.net = net
        self._temperature = temperature

        garbage_chances = [
            garbage_chance_min
            + (garbage_chance_max - garbage_chance_min) * i / max(num_envs - 1, 1)
            for i in range(num_envs)
        ]
        constructors = [
            lambda idx=i: PyTetrisEnv(
                queue_size=queue_size,
                max_holes=max_holes,
                max_height=max_height,
                max_steps=max_steps,
                max_len=max_len,
                pathfinding=True,
                seed=seed,
                idx=idx,
                garbage_chance=garbage_chances[idx],
                garbage_min=garbage_rows_min,
                garbage_max=garbage_rows_max,
                num_row_tiers=num_row_tiers,
                placement_candidates=True,
            )
            for i in range(num_envs)
        ]
        ppy_env = ParallelPyEnvironment(
            constructors, start_serially=True, blocking=False
        )
        self.env = TFPyEnvironment(ppy_env)

    def _build_candidates(self, cand_scores, cand_rows, cand_seqs, pieces):
        """Per-env: dense obs candidates -> (placements[N,128,18], mask[N,128],
        sequences[N,128,max_len])."""
        pls, masks, seqs = [], [], []
        for e in range(self._num_envs):
            valid = np.flatnonzero(cand_scores[e] > -1e29)
            pl, mask, sq = build_placement_inference(
                valid,
                cand_scores[e][valid],
                cand_rows[e][valid],
                cand_seqs[e][valid],
                active_piece=int(pieces[e][0]),
                hold_piece=int(pieces[e][1]),
                queue0=int(pieces[e][2]),
                row_norm=ROW_NORM,
                max_len=self._max_len,
            )
            pls.append(pl)
            masks.append(mask)
            seqs.append(sq)
        return np.stack(pls), np.stack(masks), np.stack(seqs)

    def collect_trajectory(self):
        b = {
            k: []
            for k in (
                "boards",
                "pieces",
                "bcg",
                "cand_placements",
                "cand_mask",
                "action_index",
                "log_prob",
                "values",
                "attacks",
                "clears",
                "attack_reward",
                "total_reward",
                "garbage_pushed",
                "dones",
            )
        }
        time_step = self.env.reset()
        for _ in range(self._num_steps):
            board = time_step.observation["board"]
            pieces = time_step.observation["pieces"]
            bcg = time_step.observation["b2b_combo_garbage"]
            cand_placements, cand_mask, cand_seqs128 = self._build_candidates(
                time_step.observation["cand_scores"].numpy(),
                time_step.observation["cand_landing_rows"].numpy(),
                time_step.observation["cand_sequences"].numpy(),
                pieces.numpy(),
            )
            cand_mask_tf = tf.constant(cand_mask)
            logits, values = self.net(
                (
                    board,
                    pieces,
                    bcg,
                    tf.constant(cand_placements, tf.float32),
                    cand_mask_tf,
                ),
                training=False,
            )
            masked = tf.where(
                cand_mask_tf, logits / self._temperature, tf.constant(-1e9, tf.float32)
            )
            dist = distributions.Categorical(logits=masked, dtype=tf.int64)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
            sel = cand_seqs128[np.arange(self._num_envs), action_index.numpy()]

            time_step = self.env.step(tf.constant(sel, tf.int64))
            reward = time_step.reward

            b["boards"].append(board)
            b["pieces"].append(pieces)
            b["bcg"].append(bcg)
            b["cand_placements"].append(tf.constant(cand_placements, tf.float32))
            b["cand_mask"].append(cand_mask_tf)
            b["action_index"].append(action_index)
            b["log_prob"].append(log_prob)
            b["values"].append(values)
            b["attacks"].append(reward["attack"])
            b["clears"].append(reward["clear"])
            b["attack_reward"].append(reward["attack_reward"])
            b["total_reward"].append(reward["total_reward"])
            b["garbage_pushed"].append(reward["garbage_pushed"][..., None])
            b["dones"].append(tf.cast(time_step.is_last(), tf.float32)[..., None])

        # bootstrap last value (state-only; no candidates needed)
        piece_dec, _ = self.net.process_obs(
            (
                time_step.observation["board"],
                time_step.observation["pieces"],
                time_step.observation["b2b_combo_garbage"],
            ),
            training=False,
        )
        last_values = self.net.score_value(piece_dec, training=False)

        out = {k: tf.stack(v) for k, v in b.items()}  # (T, N, ...)
        out["last_values"] = last_values
        return out
