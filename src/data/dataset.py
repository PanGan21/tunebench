"""Dataset loading, instruction formatting, and tokenization."""

from pathlib import Path

from datasets import Dataset, load_dataset

from tunebench.utils import ensure_pad_token

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"


def load_instruction_dataset(path: str | Path) -> Dataset:
    """Load a JSON dataset with 'instruction' and 'output' keys.

    Expects a JSON file with a list of objects:
    [{"instruction": "...", "output": "..."}, ...]
    """
    path = Path(path)
    if not path.suffix.lower() == ".json":
        raise ValueError("Dataset must be a JSON file")
    data = load_dataset("json", data_files=str(path), split="train")
    if "instruction" not in data.column_names or "output" not in data.column_names:
        raise ValueError("JSON must contain 'instruction' and 'output' columns")
    return data


def format_instruction_example(example: dict) -> dict:
    """Format a single example as ### Instruction: ... ### Response: ..."""
    text = INSTRUCTION_TEMPLATE.format(
        instruction=example["instruction"],
        output=example["output"],
    )
    return {"text": text}


def prepare_dataset(data: Dataset) -> Dataset:
    """Add formatted 'text' column and drop instruction/output."""
    return data.map(
        format_instruction_example,
        remove_columns=["instruction", "output"],
    )


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize 'text' column and add labels for causal LM (padding positions = -100)."""
    ensure_pad_token(tokenizer)
    pad_id = tokenizer.pad_token_id

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        labels = []
        for ids in out["input_ids"]:
            labels.append([x if x != pad_id else -100 for x in ids])
        out["labels"] = labels
        return out

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
