"""Pure-numpy search/value diagnostics for 1v1 AZ logging; no TF, unit-tested."""

import numpy as np


def perplexity(dist):
    """exp(H) per row = effective support size; rows should sum to 1."""
    nz = np.where(dist > 0.0, dist, 1.0)  # 0*log(0) -> 0
    return np.exp(-(dist * np.log(nz)).sum(axis=-1))


def visit_metrics(pi, cand_mask, visits):
    """Root-visit exploration means over rows with >=1 visit (None if none).

    mcts_result falls back to the prior on a 0-visit root, which would inflate
    coverage/perplexity toward "fully explored"; those rows are excluded.
    """
    searched = visits.sum(axis=1) > 0
    if not searched.any():
        return None
    p = pi[searched]
    legal = np.maximum(cand_mask[searched].sum(axis=1), 1)
    return {
        "visit_perplexity": float(perplexity(p).mean()),
        "top1_visit_share": float(p.max(axis=1).mean()),
        "top2_visit_share": float(np.partition(p, -2, axis=1)[:, -2].mean()),
        "visit_coverage": float(((p > 0.0).sum(axis=1) / legal).mean()),
    }
