"""Tests for tunebench.models.lora."""

import torch

from models import apply_lora
from tunebench.utils import count_parameters


def test_count_parameters_simple_module():
    """Count params on a simple module."""
    linear = torch.nn.Linear(10, 5)
    trainable, total, pct = count_parameters(linear)
    assert total == 10 * 5 + 5  # weight + bias
    assert trainable == total
    assert pct == 100.0


def test_count_parameters_partially_frozen():
    """Frozen params are not counted as trainable."""
    linear = torch.nn.Linear(2, 3)
    linear.weight.requires_grad = False
    trainable, total, pct = count_parameters(linear)
    assert total == 2 * 3 + 3
    assert trainable == 3  # only bias
    assert 0 < pct < 100


def test_apply_lora_reduces_trainable_params():
    """LoRA should yield far fewer trainable than total parameters."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    full_trainable, full_total, _ = count_parameters(model)
    assert full_trainable == full_total

    model = apply_lora(model, r=4)
    lora_trainable, lora_total, lora_pct = count_parameters(model)
    # LoRA adds adapter params, so total can be slightly higher; trainable is much lower
    assert lora_trainable < full_trainable
    assert lora_pct < 10.0  # LoRA typically < 1–5% trainable
