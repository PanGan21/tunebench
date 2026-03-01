"""Tests for tunebench.utils."""

from tunebench.utils import get_layer_index, get_num_transformer_layers


def test_get_num_transformer_layers_distilgpt2():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    n = get_num_transformer_layers(model)
    assert n == 6  # DistilGPT-2 has 6 layers


def test_get_layer_index():
    assert get_layer_index("transformer.h.0.attn.c_attn.weight") == 0
    assert get_layer_index("model.layers.5.self_attn.q_proj.weight") == 5
    assert get_layer_index("embed_tokens.weight") is None
