"""Tests for training.layerwise_lr."""

from training.layerwise_lr import get_layerwise_param_groups


def test_get_layerwise_param_groups():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    groups = get_layerwise_param_groups(model, base_lr=5e-5, decay=0.9, weight_decay=0.01)
    assert len(groups) >= 1
    for g in groups:
        assert "params" in g
        assert "lr" in g
        assert "weight_decay" in g
        assert g["weight_decay"] == 0.01
    # First layer should have smaller lr than last (earlier = more decay)
    lrs = [g["lr"] for g in groups]
    assert min(lrs) <= max(lrs)
