import pytest

from qtris.training.elo import EloBook


def _book() -> EloBook:
    return EloBook(init=1500.0, k_learner=2.0, k_opp=0.5, draw_weight=0.5)


def test_expected_monotonic_and_symmetric():
    book = _book()
    assert book.expected(1600.0, 1400.0) > 0.5
    # The two sides of a pairing sum to 1, so the lower-rated side is implied < 0.5.
    assert book.expected(1600.0, 1400.0) + book.expected(
        1400.0, 1600.0
    ) == pytest.approx(1.0)
    # Equal ratings -> 0.5 is also covered implicitly by test_update_closed_form_step.


def test_update_closed_form_step():
    """Equal ratings (E=0.5), a clean sweep of n=10 wins vs a non-anchor opponent:
    learner moves +k_learner*(n - n*0.5), opponent moves -k_opp*(n*0.5)."""
    book = _book()
    book.update("gen_5", wins=10, losses=0, draws=0)
    assert book.ratings["learner"] == pytest.approx(1500.0 + 2.0 * (10 - 10 * 0.5))
    assert book.ratings["gen_5"] == pytest.approx(1500.0 - 0.5 * (10 * 0.5))


def test_draw_weight_equals_win_plus_loss():
    """A draw is worth half: two draws move ratings identically to one win + one loss (both
    give s = 1.0 over n = 2). Uses a stronger opponent so the move is non-zero, which a
    degenerate equal-ratings draw (delta 0) could not distinguish from ignoring draws."""
    drawn = _book()
    drawn.ratings["gen_5"] = 1700.0
    drawn.update("gen_5", wins=0, losses=0, draws=2)

    split = _book()
    split.ratings["gen_5"] = 1700.0
    split.update("gen_5", wins=1, losses=1, draws=0)

    assert drawn.ratings["learner"] == pytest.approx(split.ratings["learner"])
    assert drawn.ratings["gen_5"] == pytest.approx(split.ratings["gen_5"])
    # Drawing twice vs a stronger opponent is an over-performance, so the learner gains.
    assert drawn.ratings["learner"] > 1500.0
    assert drawn.ratings["gen_5"] < 1700.0


def test_update_noop_on_zero_games():
    book = _book()
    book.update("gen_5", wins=0, losses=0, draws=0)
    assert book.ratings == {"learner": 1500.0, "gen_0": 1500.0}


def test_anchor_never_moves():
    book = _book()
    for _ in range(5):
        book.update("gen_0", wins=8, losses=0, draws=0)
    assert book.ratings["gen_0"] == 1500.0  # anchor is pinned
    assert book.ratings["learner"] > 1500.0  # learner still rises vs the anchor


def test_learner_rises_opponent_falls_monotonic():
    book = _book()
    last_l, last_o = book.ratings["learner"], 1500.0
    for _ in range(10):
        book.update("gen_5", wins=5, losses=0, draws=0)
        assert book.ratings["learner"] > last_l
        assert book.ratings["gen_5"] < last_o
        last_l, last_o = book.ratings["learner"], book.ratings["gen_5"]


def test_on_snapshot_inherits_learner():
    book = _book()
    book.update("gen_5", wins=12, losses=4, draws=0)
    book.on_snapshot("gen_20")
    assert book.ratings["gen_20"] == book.ratings["learner"]


def test_present_summary():
    book = _book()
    book.update("gen_5", wins=12, losses=4)  # learner climbs above gen_5
    book.on_snapshot("gen_20")  # inherits the (higher) learner rating
    summ = book.present_summary(["gen_0", "gen_5", "gen_20"])
    # best_pool excludes the learner and the gen_0 anchor.
    assert summ["best_pool"] == max(book.ratings["gen_5"], book.ratings["gen_20"])
    assert summ["learner_minus_ref"] == pytest.approx(
        book.ratings["learner"] - book.ratings["gen_0"]
    )
    assert summ["gap_to_pool"] == pytest.approx(
        book.ratings["learner"] - summ["best_pool"]
    )


def test_present_summary_empty_pool_falls_back_to_learner():
    book = _book()
    summ = book.present_summary(["gen_0"])  # only the anchor present, no real snapshots
    assert summ["best_pool"] == book.ratings["learner"]
    assert summ["gap_to_pool"] == 0.0


def test_json_roundtrip(tmp_path):
    book = _book()
    book.update("gen_5", wins=9, losses=7, draws=2)
    book.on_snapshot("gen_20")
    path = tmp_path / "elo.json"
    book.to_json(str(path))

    restored = EloBook.from_json(str(path))
    assert restored.ratings == book.ratings
    assert restored.games == book.games
    assert restored.k_learner == book.k_learner
    assert restored.anchor == book.anchor
