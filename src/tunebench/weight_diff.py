"""Weight drift: ||W_original - W_finetuned|| per layer."""

import re
from collections import defaultdict

import torch

# Same pattern as freeze_utils: transformer blocks .h.0. or .layers.0.
_LAYER_PATTERN = re.compile(r"\.(?:h|layers)\.(\d+)\.")


def _layer_key(name: str) -> str:
    """Return a group key for the param: 'layer_0', 'layer_1', ... or 'embeddings'."""
    m = _LAYER_PATTERN.search(name)
    if m is not None:
        return f"layer_{m.group(1)}"
    if "wte" in name or "wpe" in name or "embed_tokens" in name or "embed_positions" in name:
        return "embeddings"
    return "other"


def compute_weight_drift_per_layer(
    original_state_dict: dict[str, torch.Tensor],
    finetuned_state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute sum of Frobenius norms ||W_orig - W_ft|| per layer.

    Only compares parameters that exist in both dicts with the same shape.
    Keys are 'layer_0', 'layer_1', ... (transformer blocks), 'embeddings', 'other'.

    Returns:
        Dict mapping layer/group name to total drift (sum of Frobenius norms of differences).
    """
    drift: dict[str, float] = defaultdict(float)
    for name, ft_param in finetuned_state_dict.items():
        if name not in original_state_dict:
            continue
        orig_param = original_state_dict[name]
        if orig_param.shape != ft_param.shape:
            continue
        diff = (orig_param.detach().float() - ft_param.detach().float()).norm(p="fro").item()
        key = _layer_key(name)
        drift[key] += diff
    return dict(drift)


def get_state_dict_for_diff(model) -> dict[str, torch.Tensor]:
    """Get a flat state_dict suitable for diff (strip PEFT prefix if present)."""
    state = model.state_dict()
    # PEFT wraps base model: keys like base_model.model.transformer.wte.weight
    prefix = "base_model.model."
    if any(k.startswith(prefix) for k in state):
        return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    return state


def compute_weight_drift(original_model, finetuned_model) -> dict[str, float]:
    """Compute weight drift per layer between original and fine-tuned model.

    Use after full fine-tuning; for LoRA, base weights are unchanged so drift is ~0 for base.
    """
    orig_sd = get_state_dict_for_diff(original_model)
    ft_sd = get_state_dict_for_diff(finetuned_model)
    return compute_weight_drift_per_layer(orig_sd, ft_sd)


def format_drift_table(drift_per_layer: dict[str, float]) -> str:
    """Format drift dict as a table: Layer 0: 0.002, Layer 1: 0.008, ..."""
    lines = []

    # Sort so layer_0, layer_1, ... (numeric) then embeddings, other
    def sort_key(item: tuple[str, float]) -> tuple[int, int | float, str]:
        k, _ = item
        if k.startswith("layer_"):
            try:
                n = int(k.replace("layer_", ""))
                return (0, n, k)
            except ValueError:
                return (0, 0, k)
        if k == "embeddings":
            return (1, 0, k)
        return (2, 0, k)

    for key, value in sorted(drift_per_layer.items(), key=sort_key):
        if key.startswith("layer_"):
            num = key.replace("layer_", "")
            lines.append(f"Layer {num}: {value:.6f}")
        else:
            lines.append(f"{key}: {value:.6f}")
    return "\n".join(lines) if lines else "(no matching parameters)"
