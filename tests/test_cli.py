"""Tests for tunebench CLI (parsing and validation, no training)."""

import sys
from unittest.mock import patch

import pytest

from tunebench.cli import main


def test_no_subcommand_exits():
    with patch.object(sys, "argv", ["tunebench"]):
        with pytest.raises(SystemExit):
            main()


def test_weight_diff_requires_finetuned_path():
    argv = ["tunebench", "weight-diff", "--model", "distilgpt2", "--finetuned", "/nonexistent"]
    with patch.object(sys, "argv", argv):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            main()


def test_train_unknown_dataset_raises_file_not_found():
    with patch.object(
        sys,
        "argv",
        [
            "tunebench",
            "train",
            "--model",
            "distilgpt2",
            "--dataset",
            "/nonexistent/data.json",
            "--method",
            "full",
        ],
    ):
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            main()
