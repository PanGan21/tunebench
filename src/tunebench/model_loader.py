"""Base LLM loading via Hugging Face Transformers."""

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


def load_model_and_tokenizer(
    model_name: str,
    *,
    device_map: str | None = "auto",
    torch_dtype: str | None = None,
):
    """Load a causal LM and tokenizer by name (e.g. tinyllama, distilgpt2, mistral)."""
    model_id = get_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer
