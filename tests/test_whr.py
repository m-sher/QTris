import json
import math

import numpy as np
import pytest

from qtris.training.whr import SCALE, GameLog, WHRBook


def _book(tmp_path, **kw):
    return WHRBook(str(tmp_path / "games.jsonl"), **kw)


def _sim_games(rng, r1, r2, n):
    """Decisive (wins, losses) for n games between true Elo ratings r1, r2."""
    p = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
    wins = int(rng.binomial(n, p))
    return wins, n - wins


def test_synthetic_recovery(tmp_path):
    """A stationary learner 200 Elo above the anchor is recovered from eval games."""
    rng = np.random.default_rng(0)
    book = _book(tmp_path)
    for gen in range(60):
        w, lo = _sim_games(rng, 1700.0, 1500.0, 30)
        book.record(gen, "gen_0", w, lo, ctx="eval")
    book.fit(gen=59)
    assert book.converged
    assert book.ratings["learner"] == pytest.approx(1700.0, abs=40.0)
    assert book.sigmas["learner"] < 60.0


def test_anchor_exact_with_zero_sigma(tmp_path):
    book = _book(tmp_path)
    book.record(0, "gen_0", 8, 0, ctx="eval")
    book.fit()
    assert book.ratings["gen_0"] == 1500.0
    assert book.sigmas["gen_0"] == 0.0


def test_improving_trajectory_recovered(tmp_path):
    """Learner climbing ~6 Elo/gen (under the drift tracking ceiling) fits an
    increasing trajectory: late rating above early rating."""
    rng = np.random.default_rng(1)
    book = _book(tmp_path)
    for gen in range(80):
        true = 1500.0 + 6.0 * gen
        w, lo = _sim_games(rng, true, 1500.0, 30)
        book.record(gen, "gen_0", w, lo, ctx="eval")
    book.fit(gen=79)
    traj = {g: r for g, r, _s in book._trajectory}
    assert traj[79] > traj[10] + 150.0
    assert book.ratings["learner"] > 1750.0


def test_star_inflation_regression(tmp_path):
    """The motivating bug: ~50% vs recent snapshots + 37.5% vs the anchor must fit
    BELOW init, at any tie_sigma (not an artifact of the tie constant)."""
    for tie in (40.0, 100.0, 250.0):
        rng = np.random.default_rng(2)
        book = WHRBook(str(tmp_path / f"games_{int(tie)}.jsonl"), tie_sigma=tie)
        for gen in range(50):
            opp = f"gen_{10 * max(1, (gen // 10))}"
            w, lo = _sim_games(rng, 1450.0, 1450.0, 16)  # ~50% vs pool
            book.record(gen, opp, w, lo, ctx="pool")
            if gen % 10 == 0:
                w, lo = _sim_games(rng, 1450.0, 1500.0, 8)  # weaker than anchor
                book.record(gen, "gen_0", w, lo, ctx="eval")
        book.fit(gen=49)
        assert book.converged
        assert book.ratings["learner"] < 1500.0, f"inflated at tie_sigma={tie}"


def test_two_draws_equal_win_plus_loss(tmp_path):
    drawn = _book(tmp_path / "a")
    drawn.record(0, "gen_0", 0, 0, draws=2, ctx="eval")
    drawn.fit()
    split = _book(tmp_path / "b")
    split.record(0, "gen_0", 1, 1, draws=0, ctx="eval")
    split.fit()
    assert drawn.ratings["learner"] == pytest.approx(split.ratings["learner"], abs=1e-6)


def test_sigma_shrinks_with_more_games(tmp_path):
    rng = np.random.default_rng(3)
    few = _book(tmp_path / "few")
    many = _book(tmp_path / "many")
    for gen in range(5):
        w, lo = _sim_games(rng, 1600.0, 1500.0, 8)
        few.record(gen, "gen_0", w, lo, ctx="eval")
    for gen in range(50):
        w, lo = _sim_games(rng, 1600.0, 1500.0, 30)
        many.record(gen, "gen_0", w, lo, ctx="eval")
    few.fit()
    many.fit()
    assert many.sigmas["learner"] < few.sigmas["learner"]


def test_log_roundtrip_and_deterministic_refit(tmp_path):
    rng = np.random.default_rng(4)
    book = _book(tmp_path)
    for gen in range(20):
        w, lo = _sim_games(rng, 1550.0, 1500.0, 16)
        book.record(gen, "gen_10" if gen > 10 else "gen_0", w, lo, draws=1, ctx="pool")
        if gen % 10 == 0:
            book.record(gen, "gen_0", 5, 3, ctx="eval")
    book.fit(gen=19)

    fresh = WHRBook(str(tmp_path / "games.jsonl"))  # cold: reload + refit in init
    assert fresh._entries == book._entries  # ctx + draws survive the round-trip
    assert fresh.ratings["learner"] == pytest.approx(book.ratings["learner"], abs=0.01)
    assert fresh.ratings["gen_10"] == pytest.approx(book.ratings["gen_10"], abs=0.01)


def test_zero_game_snapshot_reports_tie_rating(tmp_path):
    """A registered snapshot with no games fits at the learner's rating near its
    creation gen with sigma ~ tie_sigma (the registry keeps unplayed snapshots rated)."""
    rng = np.random.default_rng(5)
    book = _book(tmp_path)
    for gen in range(30):
        true = 1500.0 + 8.0 * gen
        w, lo = _sim_games(rng, true, 1500.0, 30)
        book.record(gen, "gen_0", w, lo, ctx="eval")
    book.register_snapshot("gen_20")
    book.fit(gen=29)
    traj = {g: r for g, r, _s in book._trajectory}
    assert book.ratings["gen_20"] == pytest.approx(traj[21], abs=1.0)  # ties to k+1
    assert 30.0 < book.sigmas["gen_20"] < 150.0
    summ = book.present_summary(["gen_0", "gen_20"])
    assert summ["best_pool"] == book.ratings["gen_20"]


def test_anchor_free_log_fits_finite(tmp_path):
    """Zero anchor games (all-draw evals / truncated log): level prior keeps the
    fit well-posed instead of a singular Hessian."""
    book = _book(tmp_path)
    for gen in range(10):
        book.record(gen, "gen_5", 8, 8, ctx="pool")
    book.fit()
    assert book.converged
    assert math.isfinite(book.ratings["learner"])
    assert math.isfinite(book.sigmas["learner"])


def test_saturation_bounded(tmp_path):
    """All-wins vs everything: MAP stays finite and bounded by the priors."""
    book = _book(tmp_path)
    for gen in range(50):
        book.record(gen, "gen_0", 30, 0, ctx="pool")
    book.fit()
    assert book.converged
    assert 1500.0 < book.ratings["learner"] < 3000.0

    lost = _book(tmp_path / "lost")
    lost.record(0, "gen_0", 0, 8, ctx="eval")
    lost.fit()
    assert lost.converged and math.isfinite(lost.ratings["learner"])


def test_empty_log_fit(tmp_path):
    book = _book(tmp_path, pool_dir=None)
    book.register_snapshot("gen_10")
    book.fit()
    assert book.ratings == {"learner": 1500.0, "gen_0": 1500.0, "gen_10": 1500.0}
    assert book.sigmas["gen_0"] == 0.0


def test_record_noop_on_zero_games(tmp_path):
    book = _book(tmp_path)
    book.record(0, "gen_0", 0, 0, 0)
    assert book._entries == []
    assert book.last_gen == -1


def test_ctx_offset_tracks_eval_context(tmp_path):
    """Temp-context pool games at 45% vs anchor + greedy eval games at 55% must
    report the eval-context level (above anchor) with a negative pool offset."""
    rng = np.random.default_rng(6)
    book = _book(tmp_path)
    for gen in range(120):
        w, lo = _sim_games(rng, 1465.0, 1500.0, 30)  # ~45% temp-context
        book.record(gen, "gen_0", w, lo, ctx="pool")
        w, lo = _sim_games(rng, 1535.0, 1500.0, 8)  # ~55% greedy eval
        book.record(gen, "gen_0", w, lo, ctx="eval")
    book.fit(gen=119)
    assert book.converged
    assert book.ctx_offset < 0.0
    assert book.ratings["learner"] > 1500.0


def test_ctx_offset_prior_bounded_without_eval_games(tmp_path):
    """Biased pool-only data: without the ctx prior nearly all of the ~190 Elo
    signal would land in the offset; the prior must keep it near 0."""
    book = _book(tmp_path)
    for gen in range(20):
        book.record(gen, "gen_0", 12, 4, ctx="pool")
    book.fit()
    assert abs(book.ctx_offset) < 25.0


def test_anchor_draws_pull_level_toward_anchor(tmp_path):
    """Recorded eval draws vs the anchor damp a saturated rating vs dropping them."""
    with_draws = _book(tmp_path / "d")
    without = _book(tmp_path / "n")
    for gen in range(20):
        with_draws.record(gen, "gen_0", 6, 0, draws=2, ctx="eval")
        without.record(gen, "gen_0", 6, 0, draws=0, ctx="eval")
    with_draws.fit()
    without.fit()
    assert with_draws.ratings["learner"] < without.ratings["learner"]


def test_torn_trailing_line_tolerated_and_truncated(tmp_path):
    path = tmp_path / "games.jsonl"
    good = {
        "gen": 0,
        "p1": "learner",
        "p2": "gen_0",
        "wins": 4,
        "losses": 4,
        "draws": 0,
        "ctx": "eval",
    }
    path.write_text(json.dumps(good) + "\n" + '{"gen": 1, "p1": "lear')
    log = GameLog(str(path))
    entries = log.load()
    assert entries == [good]
    assert path.read_text() == json.dumps(good) + "\n"  # truncated back to good tail


def test_torn_midfile_line_is_hard_error(tmp_path):
    path = tmp_path / "games.jsonl"
    good = {
        "gen": 0,
        "p1": "learner",
        "p2": "gen_0",
        "wins": 4,
        "losses": 4,
        "draws": 0,
        "ctx": "eval",
    }
    path.write_text('{"gen": 0, "p1"\n' + json.dumps(good) + "\n")
    with pytest.raises(json.JSONDecodeError):
        GameLog(str(path)).load()


def test_last_gen_resume_offset(tmp_path):
    book = _book(tmp_path)
    assert book.last_gen == -1
    book.record(57, "gen_0", 4, 4, ctx="eval")
    assert book.last_gen == 57
    resumed = WHRBook(str(tmp_path / "games.jsonl"))
    assert resumed.last_gen == 57


def test_pool_dir_registry(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "gen_0.index").touch()
    (pool / "gen_10.index").touch()
    book = WHRBook(str(pool / "games.jsonl"), pool_dir=str(pool))
    book.fit()
    assert "gen_10" in book.ratings  # registered from disk, rated with zero games
    assert "gen_0" in book.ratings


def test_to_json_atomic_and_shaped(tmp_path):
    book = _book(tmp_path)
    book.record(0, "gen_0", 5, 3, ctx="eval")
    book.fit(gen=0)
    out = tmp_path / "ratings.json"
    book.to_json(str(out))
    data = json.loads(out.read_text())
    assert data["fitted_at_gen"] == 0
    assert data["converged"] is True
    entry = data["ratings"]["learner"]
    assert set(entry) == {"rating", "sigma", "games"}
    assert data["ratings"]["gen_0"]["sigma"] == 0.0
    assert not (tmp_path / "ratings.json.tmp").exists()
    assert len(data["trajectory"]) == 1  # one epoch, rounded [gen, rating, sigma]
    g, r, s = data["trajectory"][0]
    bg, br, bs = book._trajectory[0]
    assert [g, r, s] == [bg, round(br, 1), round(bs, 1)]


def test_warm_matches_cold(tmp_path):
    """Incremental per-gen fits land on the same MAP as one cold refit."""
    rng = np.random.default_rng(7)
    warm = _book(tmp_path)
    for gen in range(30):
        w, lo = _sim_games(rng, 1500.0 + 5 * gen, 1500.0, 20)
        warm.record(gen, "gen_0", w, lo, ctx="eval")
        warm.fit(gen=gen)  # warm-started every gen
    cold = WHRBook(str(tmp_path / "games.jsonl"))  # one cold fit over the log
    assert cold.ratings["learner"] == pytest.approx(warm.ratings["learner"], abs=0.05)


def test_fit_failure_keeps_previous_ratings(tmp_path, monkeypatch):
    """The never-raise contract: a solver failure must not raise, must flag
    converged=False, and must leave the previous fit fully intact."""
    book = _book(tmp_path)
    book.record(0, "gen_0", 6, 2, ctx="eval")
    book.fit(gen=0)
    prev_r = dict(book.ratings)
    prev_s = dict(book.sigmas)
    book.record(1, "gen_0", 5, 3, ctx="eval")
    monkeypatch.setattr(
        np.linalg,
        "cholesky",
        lambda A: (_ for _ in ()).throw(np.linalg.LinAlgError("boom")),
    )
    book.fit(gen=1)  # must not raise
    assert book.converged is False
    assert book.ratings == prev_r
    assert book.sigmas == prev_s
    out = tmp_path / "ratings.json"
    book.to_json(str(out))
    assert json.loads(out.read_text())["converged"] is False


def test_legacy_resume_snapshots_not_glued_to_learner(tmp_path):
    """Legacy migration: snapshots created long before the first logged epoch
    (empty log, old pool on disk) must not be pinned to the resumed learner.
    The distance-scaled tie keeps their sigma honest and lets games dominate."""
    rng = np.random.default_rng(9)
    book = _book(tmp_path)
    book.register_snapshot("gen_10")  # ~280 gens before the first epoch
    for gen in range(291, 301):
        w, lo = _sim_games(rng, 1700.0, 1580.0, 30)  # learner ~120 above gen_10
        book.record(gen, "gen_10", w, lo, ctx="pool")
        w, lo = _sim_games(rng, 1700.0, 1500.0, 8)
        book.record(gen, "gen_0", w, lo, ctx="eval")
    book.fit(gen=300)
    assert book.converged
    gap = book.ratings["learner"] - book.ratings["gen_10"]
    assert gap > 60.0, f"legacy snapshot glued to learner (gap {gap:.0f})"
    # A same-vintage zero-game snapshot keeps a wide, honest sigma.
    book.register_snapshot("gen_20")
    book.fit(gen=300)
    assert book.sigmas["gen_20"] > 120.0


def test_warm_matches_cold_with_midrun_snapshot(tmp_path):
    """Warm-start path where a NEW snapshot appears mid-sequence must land on
    the same MAP as one cold refit."""
    rng = np.random.default_rng(10)
    warm = _book(tmp_path)
    for gen in range(30):
        opp = "gen_15" if gen >= 15 else "gen_0"
        w, lo = _sim_games(rng, 1550.0, 1500.0, 16)
        warm.record(gen, opp, w, lo, ctx="pool")
        if gen % 10 == 0:
            warm.record(gen, "gen_0", 5, 3, ctx="eval")
        warm.fit(gen=gen)
    cold = WHRBook(str(tmp_path / "games.jsonl"))
    assert cold.ratings["learner"] == pytest.approx(warm.ratings["learner"], abs=0.05)
    assert cold.ratings["gen_15"] == pytest.approx(warm.ratings["gen_15"], abs=0.05)


def test_present_summary_unknown_id_and_empty_pool(tmp_path):
    book = _book(tmp_path)
    book.record(0, "gen_0", 5, 3, ctx="eval")
    book.fit(gen=0)
    summ = book.present_summary(["gen_0", "gen_999"])  # unrated id is ignored
    assert summ["best_pool"] == book.ratings["learner"]  # empty-pool fallback
    assert summ["gap_to_pool"] == 0.0


def test_pivot_sigma_matches_full_inverse(tmp_path):
    """The O(1) last-Cholesky-pivot learner sigma (full_sigma=False path) must
    equal the full inverse-Hessian marginal, which requires the latest epoch to
    be the last parameter in the layout."""
    rng = np.random.default_rng(8)
    book = _book(tmp_path)
    for gen in range(25):
        w, lo = _sim_games(rng, 1580.0, 1500.0, 20)
        book.record(gen, "gen_10" if gen > 10 else "gen_0", w, lo, ctx="pool")
        if gen % 5 == 0:
            book.record(gen, "gen_0", 4, 3, 1, ctx="eval")
    book.fit(full_sigma=True)
    full = book.sigmas["learner"]
    book.fit(full_sigma=False)
    assert book.sigmas["learner"] == pytest.approx(full, rel=1e-6)


def test_scale_constant():
    assert SCALE == pytest.approx(400.0 / math.log(10.0))
