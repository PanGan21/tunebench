"""LoRA setup and config via PEFT."""

from typing import Literal, cast

from peft import LoraConfig, PeftModel, TaskType, get_peft_model


def apply_lora(
    model,
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: Literal["none", "all", "lora_only"] = "none",
) -> PeftModel:
    """Wrap model with LoRA adapters using PEFT.

    LoRA learns a low-rank update ΔW = A @ B instead of updating the full weight matrix W,
    so fewer parameters are trained and less memory is used.

    Args:
        model: Hugging Face causal LM (e.g. from AutoModelForCausalLM).
        r: LoRA rank (dimension of A and B). Lower = fewer params; higher = more capacity.
        lora_alpha: Scaling factor (often 2*r). Larger = stronger LoRA effect.
        lora_dropout: Dropout applied in LoRA layers.
        target_modules: Module names to apply LoRA to (e.g. ["q_proj", "v_proj"]).
            If None, PEFT picks common attention projection names for the model.
        bias: Bias training ("none", "all", "lora_only").

    Returns:
        PeftModel wrapping the original model with LoRA adapters.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )
    return cast(PeftModel, get_peft_model(model, config))


def count_parameters(model) -> tuple[int, int, float]:
    """Return (trainable, total, trainable_pct) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    return trainable, total, pct
