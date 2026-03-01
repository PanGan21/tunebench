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
from training.layerwise_lr import get_layerwise_param_groups
from tunebench.utils import count_parameters

# High max_grad_norm so we don't clip but Trainer still computes and logs grad_norm
TRACK_GRAD_NORM_CLIP = 1e7


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


class TunebenchTrainer(Trainer):
    """Trainer that supports layer-wise LR decay via custom optimizer."""

    def __init__(self, *args, layer_wise_lr_decay: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_wise_lr_decay = layer_wise_lr_decay

    def create_optimizer(self):
        if self._layer_wise_lr_decay is not None and self.optimizer is None:
            import torch.optim as optim

            opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
            param_groups = get_layerwise_param_groups(
                opt_model,
                base_lr=self.args.learning_rate,
                decay=self._layer_wise_lr_decay,
                weight_decay=self.args.weight_decay,
            )
            self.optimizer = optim.AdamW(param_groups)
            return self.optimizer
        return super().create_optimizer()


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
    track_gradient_norm: bool = False,
    layer_wise_lr_decay: float | None = None,
):
    """Run fine-tuning with the Hugging Face Trainer.

    Supports full fine-tuning, LoRA, freezing, gradient norm logging,
    and layer-wise LR decay.
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

    training_kwargs: dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "logging_steps": logging_steps,
        "eval_strategy": "epoch" if eval_dataset is not None else "no",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": eval_dataset is not None,
        "report_to": "none",
    }
    if track_gradient_norm:
        training_kwargs["max_grad_norm"] = TRACK_GRAD_NORM_CLIP
    training_args = TrainingArguments(**training_kwargs)

    trainer_cls = TunebenchTrainer if layer_wise_lr_decay is not None else Trainer
    trainer_kwargs: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "callbacks": [
            PerplexityLoggingCallback(),
            ParamCountLoggingCallback(),
            MemoryAndTimeCallback(),
        ],
    }
    if layer_wise_lr_decay is not None:
        trainer_kwargs["layer_wise_lr_decay"] = layer_wise_lr_decay

    trainer = trainer_cls(**trainer_kwargs)
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
