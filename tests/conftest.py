"""Pytest fixtures and config."""

import json
from pathlib import Path

import pytest


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
