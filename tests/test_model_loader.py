"""Tests for tunebench.model_loader (no real model loading)."""

from tunebench.model_loader import MODEL_IDS, get_model_id


def test_get_model_id_tinyllama():
    assert get_model_id("tinyllama") == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_get_model_id_distilgpt2():
    assert get_model_id("distilgpt2") == "distilgpt2"


def test_get_model_id_mistral():
    assert get_model_id("mistral") == "mistralai/Mistral-7B-v0.1"


def test_get_model_id_case_insensitive():
    assert get_model_id("TINYLLAMA") == get_model_id("tinyllama")
    assert get_model_id("DistilGPT2") == "distilgpt2"


def test_get_model_id_strips_whitespace():
    assert get_model_id("  distilgpt2  ") == "distilgpt2"


def test_get_model_id_passthrough_unknown():
    custom = "my-org/my-model"
    assert get_model_id(custom) == custom


def test_model_ids_cover_shorthands():
    assert set(MODEL_IDS.keys()) == {"tinyllama", "distilgpt2", "mistral"}
