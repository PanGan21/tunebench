"""Sweep experiments: LoRA rank and learning rate."""

import json
from pathlib import Path

from analysis.metrics import loss_to_perplexity
from data.dataset import load_instruction_dataset, prepare_dataset
from models.loader import load_model_and_tokenizer
from training.trainer import run_train


def _clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_rank_sweep(
    model_name: str,
    dataset_path: str | Path,
    *,
    ranks: list[int] | None = None,
    output_base_dir: str | Path = "runs/rank-sweep",
    num_epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 512,
) -> list[dict]:
    """Train with multiple LoRA ranks and return metrics per run.

    Default ranks: [2, 4, 8, 16, 32, 64]. Results include final eval loss,
    perplexity, trainable params, and runtime.
    """
    if ranks is None:
        ranks = [2, 4, 8, 16, 32, 64]
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset(load_instruction_dataset(dataset_path))
    if len(data) >= 10:
        split = data.train_test_split(test_size=0.2, seed=42)
        train_data = split["train"]
        eval_data = split["test"]
    else:
        train_data = data
        eval_data = None

    results = []
    for rank in ranks:
        _clear_cuda_cache()
        run_dir = output_base_dir / f"rank-{rank}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_model_and_tokenizer(model_name)
        trainer = run_train(
            model,
            tokenizer,
            train_data,
            eval_dataset=eval_data,
            output_dir=run_dir,
            num_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            lora_rank=rank,
        )
        entry = {"rank": rank, "output_dir": str(run_dir)}
        if trainer.state.log_history:
            last = trainer.state.log_history[-1]
            if eval_data is not None:
                eval_loss = last.get("eval_loss")
                if eval_loss is not None:
                    entry["eval_loss"] = float(eval_loss)
                    entry["eval_perplexity"] = float(loss_to_perplexity(eval_loss))
            train_loss = last.get("train_loss")
            if train_loss is not None:
                entry["train_loss"] = float(train_loss)
            if "train_runtime" in last:
                entry["train_runtime_sec"] = float(last["train_runtime"])
        model_ref = trainer.model
        trainable = (
            sum(p.numel() for p in model_ref.parameters() if p.requires_grad) if model_ref else 0
        )
        entry["trainable_parameters"] = trainable
        results.append(entry)
        del model
        del trainer
        _clear_cuda_cache()

    out_file = output_base_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_lr_sweep(
    model_name: str,
    dataset_path: str | Path,
    *,
    learning_rates: list[float] | None = None,
    output_base_dir: str | Path = "runs/lr-sweep",
    num_epochs: int = 2,
    batch_size: int = 2,
    max_length: int = 512,
) -> list[dict]:
    """Train with multiple learning rates and return metrics per run.

    Default LRs: [1e-5, 3e-5, 5e-5, 1e-4]. Results include final eval loss,
    perplexity, and runtime.
    """
    if learning_rates is None:
        learning_rates = [1e-5, 3e-5, 5e-5, 1e-4]
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset(load_instruction_dataset(dataset_path))
    if len(data) >= 10:
        split = data.train_test_split(test_size=0.2, seed=42)
        train_data = split["train"]
        eval_data = split["test"]
    else:
        train_data = data
        eval_data = None

    results = []
    for lr in learning_rates:
        _clear_cuda_cache()
        label = f"{lr:.0e}".replace("-0", "-")
        run_dir = output_base_dir / f"lr-{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_model_and_tokenizer(model_name)
        trainer = run_train(
            model,
            tokenizer,
            train_data,
            eval_dataset=eval_data,
            output_dir=run_dir,
            num_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            max_length=max_length,
        )
        entry = {"learning_rate": lr, "output_dir": str(run_dir)}
        if eval_data is not None and trainer.state.log_history:
            last = trainer.state.log_history[-1]
            eval_loss = last.get("eval_loss")
            if eval_loss is not None:
                entry["eval_loss"] = float(eval_loss)
                entry["eval_perplexity"] = float(loss_to_perplexity(eval_loss))
        if trainer.state.log_history:
            last = trainer.state.log_history[-1]
            if "train_loss" in last:
                entry["train_loss"] = float(last["train_loss"])
            if "train_runtime" in last:
                entry["train_runtime_sec"] = float(last["train_runtime"])
        results.append(entry)
        del model
        del trainer
        _clear_cuda_cache()

    out_file = output_base_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    return results
