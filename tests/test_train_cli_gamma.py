import sys

import pytest

from qtris.cli.train import main


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    return main()


ONE_V_ONE = ["--mode", "1v1", "--algo", "az"]


def test_1v1_rejects_gamma(monkeypatch, capsys):
    """1v1 AZ exits 2 on --gamma, naming the flag."""
    with pytest.raises(SystemExit) as exc:
        _run([*ONE_V_ONE, "--gamma", "0.95"], monkeypatch)
    assert exc.value.code == 2
    assert "does not accept --gamma" in capsys.readouterr().err


def test_1v1_rejects_gamma_at_any_value(monkeypatch):
    """Rejection does not depend on the value."""
    with pytest.raises(SystemExit) as exc:
        _run([*ONE_V_ONE, "--gamma", "0.99"], monkeypatch)
    assert exc.value.code == 2


def test_1v1_without_gamma_reaches_dispatch(monkeypatch):
    """Omitting --gamma passes the guard; stop at the trainer import so no training runs."""
    sentinel = RuntimeError("dispatched")

    def _boom(*_a, **_kw):
        raise sentinel

    monkeypatch.setattr("tf_agents.system.multiprocessing.handle_main", _boom)
    with pytest.raises(RuntimeError) as exc:
        _run(ONE_V_ONE, monkeypatch)
    assert exc.value is sentinel
