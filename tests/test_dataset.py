"""Tests for tunebench.dataset."""

from pathlib import Path

import pytest

from tunebench.dataset import (
    INSTRUCTION_TEMPLATE,
    format_instruction_example,
    load_instruction_dataset,
    prepare_dataset,
)


def test_instruction_template():
    assert "### Instruction:" in INSTRUCTION_TEMPLATE
    assert "### Response:" in INSTRUCTION_TEMPLATE
    assert "{instruction}" in INSTRUCTION_TEMPLATE
    assert "{output}" in INSTRUCTION_TEMPLATE


def test_format_instruction_example():
    example = {"instruction": "Say hi.", "output": "Hi!"}
    out = format_instruction_example(example)
    assert "text" in out
    assert "### Instruction:" in out["text"]
    assert "Say hi." in out["text"]
    assert "### Response:" in out["text"]
    assert "Hi!" in out["text"]


def test_load_instruction_dataset(sample_instruction_json: Path):
    data = load_instruction_dataset(sample_instruction_json)
    assert data.num_rows == 2
    assert "instruction" in data.column_names
    assert "output" in data.column_names


def test_load_instruction_dataset_rejects_non_json(tmp_path: Path):
    bad = tmp_path / "data.txt"
    bad.write_text("not json")
    with pytest.raises(ValueError, match="JSON file"):
        load_instruction_dataset(bad)


def test_load_instruction_dataset_rejects_missing_columns(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('[{"wrong": "keys"}]')
    with pytest.raises(ValueError, match="instruction.*output"):
        load_instruction_dataset(bad)


def test_prepare_dataset(sample_instruction_json: Path):
    data = load_instruction_dataset(sample_instruction_json)
    prepared = prepare_dataset(data)
    assert "text" in prepared.column_names
    assert "instruction" not in prepared.column_names
    assert "output" not in prepared.column_names
    assert prepared.num_rows == 2
    assert "### Instruction:" in prepared[0]["text"]
    assert "### Response:" in prepared[0]["text"]
