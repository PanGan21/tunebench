"""Tests for tunebench CLI (parsing and validation, no training)."""

import sys
from unittest.mock import patch

import pytest

from tunebench.cli import main


def test_no_subcommand_exits():
    with patch.object(sys, "argv", ["tunebench"]):
        with pytest.raises(SystemExit):
            main()


def test_train_method_lora_raises_not_implemented(tmp_path):
    """Only --method full is supported; lora should raise NotImplementedError."""
    dataset = tmp_path / "data.json"
    dataset.write_text('[{"instruction": "x", "output": "y"}]')
    with patch.object(
        sys,
        "argv",
        [
            "tunebench",
            "train",
            "--model",
            "distilgpt2",
            "--dataset",
            str(dataset),
            "--method",
            "lora",
        ],
    ):
        with pytest.raises(NotImplementedError, match="Only --method full"):
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
