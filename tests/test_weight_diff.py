"""Tests for tunebench.analysis.weight_diff."""

import torch

from analysis.weight_diff import (
    compute_weight_drift_per_layer,
    format_drift_table,
    get_state_dict_for_diff,
)
from tunebench.utils import get_layer_key


def test_layer_key_transformer_blocks():
    assert get_layer_key("transformer.h.0.attn.c_attn.weight") == "layer_0"
    assert get_layer_key("model.layers.2.self_attn.q_proj.weight") == "layer_2"


def test_layer_key_embeddings():
    assert get_layer_key("transformer.wte.weight") == "embeddings"
    assert get_layer_key("model.embed_tokens.weight") == "embeddings"


def test_layer_key_other():
    assert get_layer_key("transformer.ln_f.weight") == "other"


def test_compute_weight_drift_identical_is_zero():
    """Same state dict -> zero drift."""
    sd = {"transformer.h.0.attn.c_attn.weight": torch.randn(3, 4)}
    drift = compute_weight_drift_per_layer(sd, sd)
    assert drift["layer_0"] == 0.0


def test_compute_weight_drift_difference_nonzero():
    """Different tensors -> non-zero drift."""
    orig = {"transformer.h.0.attn.c_attn.weight": torch.zeros(2, 3)}
    ft = {"transformer.h.0.attn.c_attn.weight": torch.ones(2, 3)}
    drift = compute_weight_drift_per_layer(orig, ft)
    assert drift["layer_0"] > 0


def test_compute_weight_drift_ignores_missing_keys():
    """Params only in finetuned are skipped."""
    orig = {}
    ft = {"transformer.h.0.attn.c_attn.weight": torch.ones(2, 3)}
    drift = compute_weight_drift_per_layer(orig, ft)
    assert drift == {}


def test_format_drift_table():
    drift = {"layer_1": 0.008, "layer_0": 0.002, "embeddings": 0.001}
    table = format_drift_table(drift)
    assert "Layer 0: 0.002000" in table
    assert "Layer 1: 0.008000" in table
    assert "embeddings: 0.001000" in table
    # Layer 0 before Layer 1
    assert table.index("Layer 0") < table.index("Layer 1")


def test_get_state_dict_for_diff_strips_peft_prefix():
    """PEFT-style keys get base_model.model. stripped."""

    class MockModel:
        def state_dict(self):
            return {
                "base_model.model.transformer.wte.weight": torch.randn(1, 2),
                "lora_A.default.weight": torch.randn(1, 2),
            }

    sd = get_state_dict_for_diff(MockModel())
    assert "transformer.wte.weight" in sd
    assert "base_model.model" not in str(sd.keys())
    assert "lora_A" not in sd  # only base keys when prefix present
