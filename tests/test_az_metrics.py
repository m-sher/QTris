"""Unit tests for the pure-numpy AZ diagnostics (qtris.training.az_metrics)."""

import numpy as np

from qtris.training.az_metrics import perplexity, visit_metrics


def test_perplexity_uniform_is_support_size():
    dist = np.zeros((1, 8), np.float32)
    dist[0, :4] = 1.0 / 4
    assert np.isclose(perplexity(dist)[0], 4.0)


def test_perplexity_delta_is_one():
    dist = np.zeros((1, 8), np.float32)
    dist[0, 3] = 1.0
    assert np.isclose(perplexity(dist)[0], 1.0)


def test_visit_metrics_uniform_row():
    pi = np.zeros((1, 20), np.float32)
    pi[0, :5] = 1.0 / 5
    cand_mask = np.zeros((1, 20), bool)
    cand_mask[0, :10] = True  # 10 legal candidates
    visits = np.zeros((1, 20), np.float32)
    visits[0, :5] = 2.0
    m = visit_metrics(pi, cand_mask, visits)
    assert np.isclose(m["visit_perplexity"], 5.0)
    assert np.isclose(m["top1_visit_share"], 1.0 / 5)
    assert np.isclose(m["top2_visit_share"], 1.0 / 5)
    assert np.isclose(m["visit_coverage"], 5.0 / 10)


def test_visit_metrics_delta_row():
    pi = np.zeros((1, 20), np.float32)
    pi[0, 7] = 1.0
    cand_mask = np.zeros((1, 20), bool)
    cand_mask[0, :4] = True
    visits = np.zeros((1, 20), np.float32)
    visits[0, 7] = 9.0
    m = visit_metrics(pi, cand_mask, visits)
    assert np.isclose(m["visit_perplexity"], 1.0)
    assert np.isclose(m["top1_visit_share"], 1.0)
    assert np.isclose(m["top2_visit_share"], 0.0)


def test_visit_metrics_excludes_zero_visit_prior_fallback_rows():
    # Row 0: real search (delta). Row 1: 0-visit prior-fallback spread over all legal.
    pi = np.zeros((2, 20), np.float32)
    pi[0, 3] = 1.0
    pi[1, :10] = 1.0 / 10  # looks fully explored but got no visits
    cand_mask = np.zeros((2, 20), bool)
    cand_mask[:, :10] = True
    visits = np.zeros((2, 20), np.float32)
    visits[0, 3] = 5.0  # row 1 has zero visits -> excluded
    m = visit_metrics(pi, cand_mask, visits)
    # Only row 0 counts: not inflated by the prior-fallback row.
    assert np.isclose(m["visit_perplexity"], 1.0)
    assert np.isclose(m["visit_coverage"], 1.0 / 10)


def test_visit_metrics_all_zero_returns_none():
    pi = np.full((2, 20), 1.0 / 20, np.float32)
    cand_mask = np.ones((2, 20), bool)
    visits = np.zeros((2, 20), np.float32)
    assert visit_metrics(pi, cand_mask, visits) is None
