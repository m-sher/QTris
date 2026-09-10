import sys

import pytest

from qtris.cli.train import main

ONE_V_ONE = ["--mode", "1v1"]


def _dispatched_args(argv, monkeypatch):
    """Parse argv and return the args handed to the 1v1 trainer, which is never run: the
    trainer entry is replaced by a recorder and the multiprocessing wrapper by a direct
    call, so the dispatch lambda reaches the recorder and stops there."""
    seen = {}

    def _record(args):
        seen["args"] = args
        raise RuntimeError("dispatched")

    monkeypatch.setattr("qtris.training._1v1_placement_az.main", _record)
    monkeypatch.setattr(
        "tf_agents.system.multiprocessing.handle_main", lambda fn, argv=None: fn(argv)
    )
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    with pytest.raises(RuntimeError, match="dispatched"):
        main()
    return seen["args"]


def test_1v1_target_flags_default_to_the_dense_root_bootstrap(monkeypatch):
    """gamma stays None at the parser so the trainer can pick its own default (0.97)."""
    args = _dispatched_args(ONE_V_ONE, monkeypatch)
    assert args.gamma is None
    assert args.w_value_attack == 0.006
    assert args.bootstrap == "root"
    assert args.n_step == 14


def test_the_terminal_only_search_target_is_still_reachable(monkeypatch):
    args = _dispatched_args(
        [
            *ONE_V_ONE,
            "--gamma",
            "1.0",
            "--w-value-attack",
            "0",
            "--bootstrap",
            "search",
        ],
        monkeypatch,
    )
    assert (args.gamma, args.w_value_attack, args.bootstrap) == (1.0, 0.0, "search")


def test_1v1_accepts_gamma_and_the_target_flags(monkeypatch):
    args = _dispatched_args(
        [
            *ONE_V_ONE,
            "--gamma",
            "0.97",
            "--w-value-attack",
            "0.006",
            "--bootstrap",
            "root",
        ],
        monkeypatch,
    )
    assert args.gamma == 0.97
    assert args.w_value_attack == 0.006
    assert args.bootstrap == "root"


def test_bootstrap_rejects_an_unknown_source(monkeypatch, capsys):
    """argparse exits 2 before dispatch; the dispatch is stubbed anyway so a regression
    here cannot start a training run."""
    monkeypatch.setattr(
        "tf_agents.system.multiprocessing.handle_main",
        lambda *_a, **_kw: pytest.fail("dispatch reached"),
    )
    monkeypatch.setattr(sys, "argv", ["train", *ONE_V_ONE, "--bootstrap", "leaf"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--bootstrap" in capsys.readouterr().err
