"""Tests for training.sweeps."""

from pathlib import Path

from training.sweeps import run_lr_sweep, run_rank_sweep


def test_rank_sweep_single_rank(sample_instruction_json: Path, tmp_path: Path):
    """One rank, one epoch: smoke test that rank sweep runs and writes results."""
    results = run_rank_sweep(
        "distilgpt2",
        sample_instruction_json,
        ranks=[4],
        output_base_dir=tmp_path / "rank-sweep",
        num_epochs=1,
        batch_size=2,
    )
    assert len(results) == 1
    assert results[0]["rank"] == 4
    assert "trainable_parameters" in results[0]
    out_file = tmp_path / "rank-sweep" / "results.json"
    assert out_file.exists()
    import json

    data = json.loads(out_file.read_text())
    assert len(data) == 1
    assert data[0]["rank"] == 4


def test_lr_sweep_single_lr(sample_instruction_json: Path, tmp_path: Path):
    """One LR, one epoch: smoke test that LR sweep runs and writes results."""
    results = run_lr_sweep(
        "distilgpt2",
        sample_instruction_json,
        learning_rates=[5e-5],
        output_base_dir=tmp_path / "lr-sweep",
        num_epochs=1,
        batch_size=2,
    )
    assert len(results) == 1
    assert results[0]["learning_rate"] == 5e-5
    out_file = tmp_path / "lr-sweep" / "results.json"
    assert out_file.exists()
