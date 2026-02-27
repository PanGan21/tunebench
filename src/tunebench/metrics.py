"""Training/validation loss, perplexity, GPU memory, time, trainable params."""

import math


def loss_to_perplexity(loss: float) -> float:
    """Perplexity = exp(loss)."""
    if loss is None or (isinstance(loss, float) and (loss != loss or loss < 0)):
        return float("nan")
    return math.exp(min(loss, 100.0))  # cap to avoid overflow
