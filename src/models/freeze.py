"""Layer freezing: embeddings and first N transformer layers."""

from tunebench.utils import get_layer_index


def freeze_embeddings(model) -> None:
    """Freeze embedding parameters (input token and position embeddings).

    Works with common architectures (GPT-2: wte, wpe; LLaMA/Mistral: embed_tokens).
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "wte" in name or "wpe" in name:
            param.requires_grad = False
        if "embed_tokens" in name or "embed_positions" in name:
            param.requires_grad = False


def freeze_first_n_layers(model, n: int) -> None:
    """Freeze the first n transformer blocks (layer indices 0 .. n-1).

    Works with transformer.h.0, transformer.h.1, ... (GPT-2) and
    model.layers.0, model.layers.1, ... (LLaMA/Mistral).
    """
    if n <= 0:
        return
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx = get_layer_index(name)
        if layer_idx is not None and layer_idx < n:
            param.requires_grad = False
