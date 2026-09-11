import sys

import pytest

from qtris.cli.train import main


@pytest.fixture
def dispatch(monkeypatch):
    captured = []
    monkeypatch.setattr("qtris.training._1v1_placement_az.main", captured.append)
    monkeypatch.setattr("qtris.training.placement_az.main", captured.append)
    monkeypatch.setattr(
        "tf_agents.system.multiprocessing.handle_main", lambda fn, argv: fn(argv)
    )
    return captured


@pytest.mark.parametrize("gamma", ["0.95", "0.99", "1.0"])
def test_1v1_accepts_gamma(monkeypatch, dispatch, gamma):
    monkeypatch.setattr(sys, "argv", ["train", "--mode", "1v1", "--gamma", gamma])
    main()
    assert dispatch[0].gamma == float(gamma)


def test_1v1_defaults_and_migration_flags(monkeypatch, dispatch):
    monkeypatch.setattr(
        sys, "argv", ["train", "--mode", "1v1", "--init-checkpoint", "old/ckpt-5"]
    )
    main()
    args = dispatch[0]
    assert args.gamma is None
    assert args.risk_threshold == 0.1 and args.risk_margin == 0.05
    assert args.init_checkpoint == "old/ckpt-5"
