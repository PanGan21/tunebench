"""Shared helpers: layer naming, tokenizer, parameter counts."""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Transformer block naming: .h.0. (GPT-2) or .layers.0. (LLaMA/Mistral)
LAYER_PATTERN = re.compile(r"\.(?:h|layers)\.(\d+)\.")


def get_layer_index(name: str) -> int | None:
    """Return transformer block index from param name, or None if not a block param."""
    m = LAYER_PATTERN.search(name)
    return int(m.group(1)) if m else None


def get_layer_key(name: str) -> str:
    """Return group key for param: 'layer_0', 'layer_1', ... or 'embeddings' or 'other'."""
    m = LAYER_PATTERN.search(name)
    if m is not None:
        return f"layer_{m.group(1)}"
    if "wte" in name or "wpe" in name or "embed_tokens" in name or "embed_positions" in name:
        return "embeddings"
    return "other"


def ensure_pad_token(tokenizer: "PreTrainedTokenizerBase") -> None:
    """Set pad_token to eos_token if not already set (e.g. GPT-2)."""
    if tokenizer.pad_token is not None:
        return
    tokenizer.pad_token = tokenizer.eos_token


def count_parameters(model) -> tuple[int, int, float]:
    """Return (trainable, total, trainable_pct) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    return trainable, total, pct


def get_num_transformer_layers(model) -> int:
    """Return the number of transformer blocks (e.g. .h.* or .layers.*) from param names."""
    max_idx = -1
    for name in model.state_dict():
        idx = get_layer_index(name)
        if idx is not None and idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0
