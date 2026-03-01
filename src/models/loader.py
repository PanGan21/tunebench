"""Base LLM loading via Hugging Face Transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "distilgpt2": "distilgpt2",
    "mistral": "mistralai/Mistral-7B-v0.1",
}


def get_model_id(name: str) -> str:
    """Resolve shorthand model name to HuggingFace model ID."""
    key = name.lower().strip()
    if key in MODEL_IDS:
        return MODEL_IDS[key]
    return name


def _default_dtype():
    """Prefer CUDA-friendly dtype when GPU is available (saves VRAM, faster)."""
    if not torch.cuda.is_available():
        return None
    # Prefer bfloat16 on Ampere+; fallback to float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(
    model_name: str,
    *,
    device_map: str | None = "auto",
    torch_dtype: str | torch.dtype | None = None,
):
    """Load a causal LM and tokenizer by name (e.g. tinyllama, distilgpt2, mistral).

    When CUDA is available, uses bfloat16/float16 by default to reduce VRAM and speed up training.
    """
    model_id = get_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch_dtype is None:
        torch_dtype = _default_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer
