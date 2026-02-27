"""Training loop and Trainer integration (full / freeze / LoRA)."""

from pathlib import Path

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from tunebench.metrics import loss_to_perplexity


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is not None:
        return
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_dataset(dataset, tokenizer, max_length: int = 512):
    """Tokenize 'text' column and add labels for causal LM (padding positions = -100)."""
    _ensure_pad_token(tokenizer)
    pad_id = tokenizer.pad_token_id

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        # Labels: same as input_ids, -100 at padding (so loss is not computed there)
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


class PerplexityLoggingCallback(TrainerCallback):
    """Callback that adds eval_perplexity = exp(eval_loss) to logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            logs["eval_perplexity"] = loss_to_perplexity(logs["eval_loss"])
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
):
    """Run full fine-tuning with the Hugging Face Trainer.

    Logs training loss, validation loss, and perplexity (exp(loss)).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        callbacks=[PerplexityLoggingCallback()],
    )

    trainer.train()
    return trainer
