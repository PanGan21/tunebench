"""CLI entry: tunebench train / weight-diff / forgetting-test."""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM

from analysis import (
    compute_head_importance,
    compute_weight_drift,
    format_drift_table,
    format_head_importance_table,
    loss_to_perplexity,
)
from data import load_instruction_dataset, prepare_dataset
from models import get_model_id, load_model_and_tokenizer
from training import evaluate_loss, run_lr_sweep, run_rank_sweep, run_train


def _train(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = load_instruction_dataset(dataset_path)
    data = prepare_dataset(data)

    # Train/validation split (80/20) if we have more than a few examples
    if len(data) >= 10 and args.validation_split > 0:
        split = data.train_test_split(test_size=args.validation_split, seed=42)
        train_data = split["train"]
        eval_data = split["test"]
    else:
        train_data = data
        eval_data = None

    model, tokenizer = load_model_and_tokenizer(args.model)

    lora_rank = args.lora_rank if args.method == "lora" else None

    run_train(
        model,
        tokenizer,
        train_data,
        eval_dataset=eval_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        lora_rank=lora_rank,
        freeze_embeddings_layer=args.freeze_embeddings,
        freeze_first_n=args.freeze_first_n_layers,
        track_gradient_norm=getattr(args, "track_gradient_norm", False),
        layer_wise_lr_decay=getattr(args, "layer_wise_lr_decay", None),
    )


def _weight_diff(args: argparse.Namespace) -> None:
    """Compare original vs fine-tuned weights per layer."""
    from peft import PeftModel

    model_id = get_model_id(args.model)
    finetuned_path = Path(args.finetuned)
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {finetuned_path}")

    original_model = AutoModelForCausalLM.from_pretrained(model_id)
    finetuned_model = AutoModelForCausalLM.from_pretrained(str(finetuned_path))
    if (finetuned_path / "adapter_config.json").exists():
        finetuned_model = PeftModel.from_pretrained(finetuned_model, str(finetuned_path))

    drift = compute_weight_drift(original_model, finetuned_model)
    print("Weight drift ||W_original - W_finetuned|| (Frobenius norm sum per layer):")
    print(format_drift_table(drift))


def _forgetting_test(args: argparse.Namespace) -> None:
    """Eval base on generic prompts, fine-tune on narrow data, re-eval; report degradation."""
    train_path = Path(args.train_dataset)
    eval_path = Path(args.eval_dataset)
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_path}")

    eval_data = prepare_dataset(load_instruction_dataset(eval_path))
    train_data = load_instruction_dataset(train_path)
    train_data = prepare_dataset(train_data)

    model, tokenizer = load_model_and_tokenizer(args.model)
    max_len = args.max_length
    batch = args.batch_size

    print("Evaluating base model on generic prompts...")
    loss_before = evaluate_loss(model, tokenizer, eval_data, max_length=max_len, batch_size=batch)
    ppl_before = loss_to_perplexity(loss_before)
    print(f"  eval_loss={loss_before:.4f}  eval_perplexity={ppl_before:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_train(
        model,
        tokenizer,
        train_data,
        eval_dataset=None,
        output_dir=output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=args.learning_rate,
        max_length=max_len,
        logging_steps=args.logging_steps,
    )

    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {output_dir}")
    ft_model = AutoModelForCausalLM.from_pretrained(str(checkpoints[-1]))

    print("Evaluating fine-tuned model on same generic prompts...")
    loss_after = evaluate_loss(ft_model, tokenizer, eval_data, max_length=max_len, batch_size=batch)
    ppl_after = loss_to_perplexity(loss_after)
    print(f"  eval_loss={loss_after:.4f}  eval_perplexity={ppl_after:.4f}")

    degradation = loss_after - loss_before
    print("\nCatastrophic forgetting (higher loss = more forgetting):")
    print(f"  loss_degradation={degradation:+.4f}")
    print(f"  (before {loss_before:.4f} -> after {loss_after:.4f})")
    print(f"  perplexity: {ppl_before:.4f} -> {ppl_after:.4f}")


def _rank_sweep(args: argparse.Namespace) -> None:
    """Run LoRA rank sweep and print summary."""
    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    results = run_rank_sweep(
        args.model,
        path,
        ranks=args.ranks,
        output_base_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    print("Rank sweep results:")
    for r in results:
        print(f"  rank={r['rank']} trainable={r.get('trainable_parameters', 'N/A')} ", end="")
        if "eval_loss" in r:
            print(
                f"eval_loss={r['eval_loss']:.4f} eval_ppl={r.get('eval_perplexity', 0):.4f}", end=""
            )
        print(f" runtime_sec={r.get('train_runtime_sec', 'N/A')}")
    print(f"Full results: {Path(args.output_dir) / 'results.json'}")


def _lr_sweep(args: argparse.Namespace) -> None:
    """Run learning rate sweep and print summary."""
    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    results = run_lr_sweep(
        args.model,
        path,
        learning_rates=args.learning_rates,
        output_base_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print("LR sweep results:")
    for r in results:
        lr = r["learning_rate"]
        print(f"  lr={lr:.0e} ", end="")
        if "eval_loss" in r:
            print(
                f"eval_loss={r['eval_loss']:.4f} eval_ppl={r.get('eval_perplexity', 0):.4f}", end=""
            )
        print(f" runtime_sec={r.get('train_runtime_sec', 'N/A')}")
    print(f"Full results: {Path(args.output_dir) / 'results.json'}")


def _head_importance(args: argparse.Namespace) -> None:
    """Compute and print attention head importance table."""
    model, _ = load_model_and_tokenizer(args.model)
    importance = compute_head_importance(model)
    print("Attention head importance (weight L2 norm per head):")
    print(format_head_importance_table(importance, max_heads=args.max_heads))


def main() -> None:
    parser = argparse.ArgumentParser(prog="tunebench", description="Fine-tune playground for LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run fine-tuning")
    train_parser.add_argument(
        "--model", required=True, help="Model: tinyllama, distilgpt2, mistral, or HF model ID"
    )
    train_parser.add_argument(
        "--dataset", required=True, help="Path to JSON dataset (instruction + output)"
    )
    train_parser.add_argument(
        "--method",
        choices=("full", "lora", "freeze"),
        default="full",
        help="Fine-tuning method (default: full)",
    )
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    train_parser.add_argument(
        "--output-dir", default="runs", help="Output directory (default: runs)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device batch size (default: 2)"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)"
    )
    train_parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length (default: 512)"
    )
    train_parser.add_argument(
        "--validation-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)"
    )
    train_parser.add_argument(
        "--logging-steps", type=int, default=1, help="Log every N steps (default: 1)"
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank when --method lora (default: 8). Ignored for full/freeze.",
    )
    train_parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding layers (saves memory, often sufficient for instruction tuning).",
    )
    train_parser.add_argument(
        "--freeze-first-n-layers",
        type=int,
        default=0,
        metavar="N",
        help="Freeze first N transformer layers (default: 0). Use with --method freeze or full.",
    )
    train_parser.add_argument(
        "--track-gradient-norm",
        action="store_true",
        help="Log gradient L2 norm each step (useful for debugging training dynamics).",
    )
    train_parser.add_argument(
        "--layer-wise-lr-decay",
        type=float,
        default=None,
        metavar="DECAY",
        help="Layer-wise LR decay (e.g. 0.9): earlier layers get smaller LR.",
    )

    # weight-diff
    wd_parser = subparsers.add_parser(
        "weight-diff", help="Compare original vs fine-tuned weights per layer"
    )
    wd_parser.add_argument("--model", required=True, help="Base model (e.g. distilgpt2) or HF ID")
    wd_parser.add_argument("--finetuned", required=True, help="Path to fine-tuned checkpoint dir")

    # forgetting-test
    ft_parser = subparsers.add_parser(
        "forgetting-test",
        help="Eval base on generic prompts, fine-tune, re-eval; report degradation",
    )
    ft_parser.add_argument("--model", required=True, help="Model name or HF ID")
    ft_parser.add_argument("--train-dataset", required=True, help="Narrow-domain train JSON")
    ft_parser.add_argument(
        "--eval-dataset", required=True, help="Generic eval JSON (instruction+output)"
    )
    ft_parser.add_argument("--output-dir", default="runs/forgetting-test", help="Checkpoint dir")
    ft_parser.add_argument("--epochs", type=int, default=3)
    ft_parser.add_argument("--batch-size", type=int, default=2)
    ft_parser.add_argument("--learning-rate", type=float, default=5e-5)
    ft_parser.add_argument("--max-length", type=int, default=512)
    ft_parser.add_argument("--logging-steps", type=int, default=1)

    # rank-sweep
    rs_parser = subparsers.add_parser(
        "rank-sweep",
        help="Train with multiple LoRA ranks (2–64), save metrics to results.json",
    )
    rs_parser.add_argument("--model", required=True, help="Model name or HF ID")
    rs_parser.add_argument("--dataset", required=True, help="Path to instruction JSON")
    rs_parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="LoRA ranks to try (default: 2 4 8 16 32 64)",
    )
    rs_parser.add_argument("--output-dir", default="runs/rank-sweep")
    rs_parser.add_argument("--epochs", type=int, default=2)
    rs_parser.add_argument("--batch-size", type=int, default=2)
    rs_parser.add_argument("--learning-rate", type=float, default=5e-5)
    rs_parser.add_argument("--max-length", type=int, default=512)

    # lr-sweep
    lr_parser = subparsers.add_parser(
        "lr-sweep",
        help="Train with multiple learning rates, save metrics to results.json",
    )
    lr_parser.add_argument("--model", required=True, help="Model name or HF ID")
    lr_parser.add_argument("--dataset", required=True, help="Path to instruction JSON")
    lr_parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 3e-5, 5e-5, 1e-4],
        help="Learning rates to try (default: 1e-5 3e-5 5e-5 1e-4)",
    )
    lr_parser.add_argument("--output-dir", default="runs/lr-sweep")
    lr_parser.add_argument("--epochs", type=int, default=2)
    lr_parser.add_argument("--batch-size", type=int, default=2)
    lr_parser.add_argument("--max-length", type=int, default=512)

    # head-importance
    hi_parser = subparsers.add_parser(
        "head-importance",
        help="Compute attention head importance (weight norm) per layer",
    )
    hi_parser.add_argument("--model", required=True, help="Model name or HF ID")
    hi_parser.add_argument(
        "--max-heads",
        type=int,
        default=12,
        help="Max heads to show per layer (default: 12)",
    )

    args = parser.parse_args()

    if args.command == "train":
        _train(args)
    elif args.command == "weight-diff":
        _weight_diff(args)
    elif args.command == "forgetting-test":
        _forgetting_test(args)
    elif args.command == "rank-sweep":
        _rank_sweep(args)
    elif args.command == "lr-sweep":
        _lr_sweep(args)
    elif args.command == "head-importance":
        _head_importance(args)
