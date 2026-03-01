"""Layer-wise learning rate decay: earlier layers get smaller LR."""

from tunebench.utils import get_layer_index


def get_layerwise_param_groups(model, base_lr: float, decay: float, weight_decay: float = 0.0):
    """Build optimizer param groups with layer-wise LR decay.

    Layer 0 gets base_lr * decay^(num_layers-1), last layer gets base_lr.
    Embeddings and other params get base_lr * decay^(num_layers).
    """
    num_layers = _count_layers(model)
    groups: dict[int, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        idx = get_layer_index(name)
        if idx is not None:
            exponent = num_layers - 1 - idx
            lr = base_lr * (decay**exponent)
        else:
            exponent = num_layers
            lr = base_lr * (decay**exponent)
        if exponent not in groups:
            groups[exponent] = {"params": [], "lr": lr, "weight_decay": weight_decay}
        groups[exponent]["params"].append(param)
    return list(groups.values())


def _count_layers(model) -> int:
    max_idx = -1
    for name in model.state_dict():
        idx = get_layer_index(name)
        if idx is not None and idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 1
