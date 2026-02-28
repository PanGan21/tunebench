"""Pytest fixtures and config."""

import json
from pathlib import Path

import pytest

# PEFT warns when applying LoRA to GPT-2/DistilGPT2 (Conv1D): it sets fan_in_fan_out=True.
# Suppress so test output stays clean; behavior is correct.
pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore::UserWarning:peft.tuners.lora.layer",
    )


@pytest.fixture
def sample_instruction_json(tmp_path: Path) -> Path:
    """Write a minimal instruction JSON to a temp file."""
    data = [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "Say hello.", "output": "Hello!"},
    ]
    path = tmp_path / "instructions.json"
    path.write_text(json.dumps(data, indent=2))
    return path
