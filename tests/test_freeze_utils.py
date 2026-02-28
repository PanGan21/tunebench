"""Tests for tunebench.freeze_utils."""

from tunebench.freeze_utils import freeze_embeddings, freeze_first_n_layers
from tunebench.lora_utils import count_parameters


def test_freeze_embeddings_reduces_trainable():
    """Freezing embeddings should reduce trainable parameter count."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    trainable_before, total, _ = count_parameters(model)

    freeze_embeddings(model)
    trainable_after, _, _ = count_parameters(model)

    assert trainable_after < trainable_before
    assert trainable_after < total


def test_freeze_first_n_layers_reduces_trainable():
    """Freezing first N layers should reduce trainable parameter count."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    trainable_before, total, _ = count_parameters(model)

    freeze_first_n_layers(model, 2)  # freeze first 2 of 6 layers
    trainable_after, _, _ = count_parameters(model)

    assert trainable_after < trainable_before
    assert trainable_after < total


def test_freeze_first_n_layers_zero_no_op():
    """freeze_first_n_layers(0) should not change anything."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    trainable_before, _, _ = count_parameters(model)

    freeze_first_n_layers(model, 0)
    trainable_after, _, _ = count_parameters(model)

    assert trainable_after == trainable_before
