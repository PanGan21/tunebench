"""Tests for analysis.head_importance."""

from analysis.head_importance import (
    compute_head_importance,
    format_head_importance_table,
)


def test_compute_head_importance_distilgpt2():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    importance = compute_head_importance(model)
    assert isinstance(importance, dict)
    # DistilGPT-2 has 6 layers
    assert len(importance) == 6
    for _, scores in importance.items():
        assert isinstance(scores, list)
        assert len(scores) == 12  # 12 heads
        assert all(isinstance(s, (int, float)) for s in scores)


def test_format_head_importance_table():
    importance = {0: [1.0, 2.0, 3.0], 1: [1.5, 2.5, 3.5]}
    table = format_head_importance_table(importance, max_heads=3)
    assert "Layer" in table
    assert "H0" in table
    assert "1.0000" in table
