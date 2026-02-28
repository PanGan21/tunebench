"""Training loop and Trainer integration (full / freeze / LoRA)."""

import tempfile
from pathlib import Path

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from analysis.metrics import loss_to_perplexity
from data.dataset import tokenize_dataset
from models.freeze import freeze_embeddings, freeze_first_n_layers
from models.lora import apply_lora
from tunebench.utils import count_parameters


class PerplexityLoggingCallback(TrainerCallback):
    """Callback that adds eval_perplexity = exp(eval_loss) to logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            logs["eval_perplexity"] = loss_to_perplexity(logs["eval_loss"])
        return control


class ParamCountLoggingCallback(TrainerCallback):
    """Log total params, trainable params, and trainable % at training start."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            trainable, total, pct = count_parameters(model)
            print(
                f"[tunebench] total_parameters={total:,} "
                f"trainable_parameters={trainable:,} "
                f"trainable_pct={pct:.2f}%"
            )
        return control


class MemoryAndTimeCallback(TrainerCallback):
    """Log peak GPU memory (if CUDA) and time per epoch at end of training."""

    def on_train_end(self, args, state, control, **kwargs):
        if state.log_history:
            last = state.log_history[-1]
            total_time = last.get("train_runtime")
            epoch = last.get("epoch")
            if total_time is not None and epoch is not None and epoch > 0:
                try:
                    t, e = float(total_time), float(epoch)
                    print(f"[tunebench] time_per_epoch_sec={t / e:.2f}")
                except (TypeError, ValueError):
                    pass
        try:
            import torch

            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"[tunebench] gpu_peak_memory_mb={peak_mb:.2f}")
        except Exception:
            pass
        return control


def run_train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    *,
    output_dir: str | Path = "runs",
    num_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    logging_steps: int = 1,
    lora_rank: int | None = None,
    freeze_embeddings_layer: bool = False,
    freeze_first_n: int = 0,
):
    """Run fine-tuning with the Hugging Face Trainer.

    Supports full fine-tuning, LoRA (when lora_rank is set), and optional
    freezing of embeddings and/or first N layers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lora_rank is not None:
        model = apply_lora(model, r=lora_rank)
    if freeze_embeddings_layer:
        freeze_embeddings(model)
    if freeze_first_n > 0:
        freeze_first_n_layers(model, freeze_first_n)

    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    if eval_dataset is not None:
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=eval_dataset is not None,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            PerplexityLoggingCallback(),
            ParamCountLoggingCallback(),
            MemoryAndTimeCallback(),
        ],
    )

    trainer.train()
    return trainer


def evaluate_loss(
    model,
    tokenizer,
    dataset,
    *,
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    """Compute average cross-entropy loss (and thus perplexity) on a dataset.

    Dataset must have a 'text' column (e.g. from prepare_dataset).
    """
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=tokenized,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()
    return float(metrics["eval_loss"])
