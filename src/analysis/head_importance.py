"""Attention head importance: weight-norm based per-layer, per-head scores."""

import re
from collections import defaultdict


def _get_num_heads_and_head_dim(model) -> tuple[int, int]:
    """Infer num_heads and head_dim from config or param shapes."""
    config = getattr(model, "config", None)
    if config is None:
        return 12, 64  # fallback
    num_heads = getattr(config, "n_head", None) or getattr(config, "num_attention_heads", 12)
    hidden = getattr(config, "n_embd", None) or getattr(config, "hidden_size", 768)
    head_dim = getattr(config, "head_dim", None) or (hidden // num_heads)
    return int(num_heads), int(head_dim)


def _layer_idx_from_name(name: str) -> int | None:
    m = re.search(r"\.(?:h|layers)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def compute_head_importance(model) -> dict[int, list[float]]:
    """Compute per-layer, per-head importance as L2 norm of that head's weights.

    Uses Q (and optionally K, V) projection weights; aggregates so each layer
    has one score per head. Returns {layer_idx: [score_head0, score_head1, ...]}.
    """
    num_heads, head_dim = _get_num_heads_and_head_dim(model)
    # Aggregate norm per (layer, head)
    layer_head_norms: dict[int, list[float]] = defaultdict(lambda: [0.0] * num_heads)
    state = model.state_dict()
    # Handle PEFT wrapper
    if any(k.startswith("base_model.model.") for k in state):
        state = {
            k.replace("base_model.model.", ""): v
            for k, v in state.items()
            if k.startswith("base_model.model.")
        }

    for name, param in state.items():
        layer_idx = _layer_idx_from_name(name)
        if layer_idx is None:
            continue
        if "attn" not in name and "self_attn" not in name:
            continue
        # Q/K/V/O projection: (out_features, in_features); split out_features by head
        if param.dim() != 2:
            continue
        out_dim, in_dim = param.shape
        # Assume out_dim = num_heads * head_dim (or 3* that for merged qkv)
        if "c_attn" in name or "qkv" in name:
            # GPT-2 style: (3*n_embd, n_embd); Q,K,V stacked
            out_per_proj = out_dim // 3
            if out_per_proj % num_heads != 0:
                continue
            per_head = out_per_proj // num_heads
            for h in range(num_heads):
                total = 0.0
                for block in range(3):
                    base = block * out_per_proj
                    start = base + h * per_head
                    end = base + (h + 1) * per_head
                    total += param.data[start:end].float().norm(p="fro").item()
                layer_head_norms[layer_idx][h] += total
        else:
            if out_dim % num_heads != 0:
                continue
            per_head = out_dim // num_heads
            for h in range(num_heads):
                start = h * per_head
                end = start + per_head
                chunk = param.data[start:end].float().norm(p="fro").item()
                layer_head_norms[layer_idx][h] += chunk

    return dict(layer_head_norms)


def format_head_importance_table(importance: dict[int, list[float]], max_heads: int = 12) -> str:
    """Format head importance as a text table (layer x head)."""
    if not importance:
        return "(no attention layers found)"
    lines = [
        "Layer\t"
        + "\t".join(f"H{i}" for i in range(min(max_heads, len(next(iter(importance.values()))))))
    ]
    for layer in sorted(importance.keys()):
        scores = importance[layer]
        row = "\t".join(f"{s:.4f}" for s in scores[:max_heads])
        lines.append(f"{layer}\t{row}")
    return "\n".join(lines)
