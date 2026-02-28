"""Layer freezing: embeddings and first N transformer layers."""

import re


def freeze_embeddings(model) -> None:
    """Freeze embedding parameters (input token and position embeddings).

    Works with common architectures (GPT-2: wte, wpe; LLaMA/Mistral: embed_tokens).
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # GPT-2 / DistilGPT-2
        if "wte" in name or "wpe" in name:
            param.requires_grad = False
        # LLaMA / Mistral / TinyLlama
        if "embed_tokens" in name or "embed_positions" in name:
            param.requires_grad = False


def freeze_first_n_layers(model, n: int) -> None:
    """Freeze the first n transformer blocks (layer indices 0 .. n-1).

    Works with transformer.h.0, transformer.h.1, ... (GPT-2) and
    model.layers.0, model.layers.1, ... (LLaMA/Mistral).
    """
    if n <= 0:
        return
    # Match .h.0. or .layers.0. etc. to get layer index
    layer_pattern = re.compile(r"\.(?:h|layers)\.(\d+)\.")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        m = layer_pattern.search(name)
        if m is not None:
            layer_idx = int(m.group(1))
            if layer_idx < n:
                param.requires_grad = False
