"""CLI entry: tunebench train --model ... --dataset ... --method full|lora|freeze --epochs ..."""

import argparse
from pathlib import Path

from tunebench.dataset import load_instruction_dataset, prepare_dataset
from tunebench.model_loader import load_model_and_tokenizer
from tunebench.trainer import run_train


def _train(args: argparse.Namespace) -> None:
    if args.method != "full":
        raise NotImplementedError(
            f"Only --method full is implemented for now. Got: {args.method}. "
            "LoRA and freeze will be added in a later phase."
        )
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
    )


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

    args = parser.parse_args()

    if args.command == "train":
        _train(args)
