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
