from pathlib import Path
import subprocess

import numpy as np
import pytest
import tensorflow as tf

from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY as CAP, RISK_HORIZON as H
from qtris.search.placement_mcts import PlacementMCTS
from qtris.training.attack_risk import attack_risk_config
from test_mcts_break_credit import _position_with_break_and_maintain
from test_mcts_rounds import _played_env


def test_c_committed_statistics_gate_and_risk_backup(tmp_path):
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "mcts_invariants"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-std=c99",
            "-fopenmp",
            str(root / "tests/mcts_invariants.c"),
            str(root / "tetrisenv/TetrisEnv/pathfinder.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


@pytest.fixture(scope="module")
def position():
    return _position_with_break_and_maintain()


class ConstantNet:
    def __init__(self, favored=None):
        self.favored = favored

    def policy_attack_risk(self, inputs):
        n = int(inputs[0].shape[0])
        logits = np.full((n, CAP), -80, np.float32)
        logits[:, self.favored if self.favored is not None else 0] = 80
        risks = np.broadcast_to(
            np.linspace(0.05 / H, 0.05, H, dtype=np.float32), (n, CAP, H)
        )
        return tf.constant(logits), tf.zeros((n, 1)), tf.constant(risks)


def test_gate_excludes_large_surge_under_noise_temperature_and_fallback(position):
    env, brk, _, _ = position
    for budget in (0, 17, 256):
        cfg = attack_risk_config(
            num_simulations=budget, leaves_per_round=8, dirichlet_eps=1
        )
        search = PlacementMCTS(ConstantNet(brk), cfg)
        row = search.search([env], 1, [1])[0]
        assert not row["break_mask"][row["slot"]]
        assert row["pi"][brk] == 0
        assert not row["gate_mask"][brk]
        assert row["visits"] == row["completed_simulations"] == budget
        assert row["pi"].sum() == pytest.approx(1)


def test_true_scorer_classifies_nonclear_setups_and_breaks(position):
    env, brk, keep, cands = position
    search = PlacementMCTS(ConstantNet(), attack_risk_config(num_simulations=0))
    row = search.search([env], 1, 0)[0]
    assert row["break_mask"][brk] and not row["break_mask"][keep]
    for slot, (clears, _attack, new_b2b) in cands.items():
        if clears == 0:
            assert row["gate_mask"][slot]
        assert row["break_mask"][slot] == (clears > 0 and new_b2b == -1)


def test_collision_budget_and_terminal_only_rounds(position):
    env, _, _, _ = position
    for max_holes in (-1, 0):
        engine = CMCTS(
            1,
            num_simulations=257,
            leaves_per_round=8,
            max_holes=max_holes,
            w_attack=0,
            w_death=0,
        )
        try:
            engine.set_root(0, env)
            n, _ = engine.collect_roots()
            logits = np.full((n, CAP), -80, np.float32)
            logits[:, 0] = 80
            engine.apply_roots(
                logits, np.zeros(n, np.float32), np.zeros_like(logits), 0
            )
            rounds = 0
            while engine.progress()[0, 0] < 257:
                n, _ = engine.collect_leaves()
                logits = np.full((n, CAP), -80, np.float32)
                logits[:, 0] = 80
                engine.apply_leaves(logits, np.zeros(n, np.float32))
                rounds += 1
                assert rounds <= 257
            assert engine.result()[1].sum() == 257
        finally:
            engine.destroy()


def test_delayed_risk_can_authorize_a_break(position):
    env, brk, _, _ = position

    class ThreatNet(ConstantNet):
        def policy_attack_risk(self, inputs):
            logits, value, _ = super().policy_attack_risk(inputs)
            n = int(inputs[0].shape[0])
            curves = np.zeros((n, CAP, H), np.float32)
            curves[:, :, 11:] = 0.30
            curves[:, brk, 11:] = 0.20
            return logits, value, tf.constant(curves)

    result = PlacementMCTS(
        ThreatNet(brk), attack_risk_config(num_simulations=0)
    ).search([env], 1, 0)[0]
    assert result["slot"] == brk
    assert result["risk_curve"][0] == 0
    assert result["risk_best_keep"] - result["risk_chosen"] >= 0.05


def test_known_immediate_death_overrides_fresh_risk_prior(position):
    env, _, _, cands = position
    previous = list(env._garbage_queue)
    try:
        env._garbage_queue = [(40, 3, 1)]
        result = PlacementMCTS(
            ConstantNet(), attack_risk_config(num_simulations=0)
        ).search([env], 1, 0)[0]
        for slot, (clears, _, _) in cands.items():
            if not clears:
                assert not result["gate_mask"][slot]
        assert cands[result["slot"]][0] > 0
    finally:
        env._garbage_queue = previous


def test_plain_clears_remain_unproductive_without_an_active_bank(position):
    env, brk, _, _ = position
    previous = env._scorer._b2b
    try:
        env._scorer._b2b = -1
        result = PlacementMCTS(
            ConstantNet(brk), attack_risk_config(num_simulations=0)
        ).search([env], 1, 0)[0]
        assert result["break_mask"][brk]
        assert not result["gate_mask"][brk]
    finally:
        env._scorer._b2b = previous


@pytest.mark.parametrize("risk24", [0.09, 0.11, 0.25])
def test_preserving_alternatives_fill_batches_above_risk_threshold(risk24):
    class VariedRiskNet:
        calls = 0

        def policy_attack_risk(self, inputs):
            self.calls += 1
            n = inputs[0].shape[0]
            curve = (risk24 + np.arange(CAP)[:, None] * 0.0001) * np.linspace(
                1 / H, 1, H
            )
            return (
                tf.zeros((n, CAP)),
                tf.zeros((n, 1)),
                tf.constant(np.broadcast_to(curve, (n, CAP, H)), tf.float32),
            )

    env = _played_env(7)
    net = VariedRiskNet()
    search = PlacementMCTS(
        net,
        attack_risk_config(num_simulations=256, leaves_per_round=8, dirichlet_eps=0),
    )
    rows = search.search([env] * 16, 1, 1)
    stats = search.last_stats
    assert net.calls == stats["inference_calls"]
    assert stats["inference_calls"] <= 40
    assert stats["inference_rows"] / stats["inference_capacity"] > 0.8
    assert stats["inference_seconds"] <= stats["seconds"]
    for row in rows:
        assert row["completed_simulations"] == row["visits"] == 256
        assert not row["break_mask"][row["slot"]]
        assert row["gate_mask"][row["slot"]]
        assert not row["pi"][~row["gate_mask"]].any()
        assert row["counts"][row["break_mask"]].sum() == 0
