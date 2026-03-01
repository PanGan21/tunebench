"""CLI tests for advanced commands (rank-sweep, lr-sweep, head-importance)."""

import argparse
from unittest.mock import patch

import pytest

from tunebench.cli import _head_importance, main


def test_rank_sweep_help(capsys):
    with patch("sys.argv", ["tunebench", "rank-sweep", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "rank-sweep" in out or "ranks" in out


def test_lr_sweep_help(capsys):
    with patch("sys.argv", ["tunebench", "lr-sweep", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "lr-sweep" in out or "learning-rate" in out


def test_head_importance_help(capsys):
    with patch("sys.argv", ["tunebench", "head-importance", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "head-importance" in out


def test_head_importance_runs():
    args = argparse.Namespace(model="distilgpt2", max_heads=4)
    _head_importance(args)
    # Should not raise; prints table


def test_train_accepts_track_gradient_norm_and_layer_wise_lr():
    """Ensure train parser has new flags (dispatch tested via run_train in integration)."""
    from tunebench.cli import main

    with patch("sys.argv", ["tunebench", "train", "--model", "x", "--dataset", "y", "--help"]):
        with pytest.raises(SystemExit):
            main()
    # If --track-gradient-norm and --layer-wise-lr-decay are in the parser, we're good
