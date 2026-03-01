# Tunebench

CLI for comparing full fine-tuning, LoRA, and layer freezing on small LLMs.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependencies and virtual environments. Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv
```

### Virtual environment

**Option A — Let uv create and use the venv (recommended)**

```bash
uv sync
```

This creates a `.venv` in the project root, installs all dependencies from the lockfile, and installs `tunebench` in editable mode. You can then:

- **Activate the venv** and run commands as usual:
  ```bash
  source .venv/bin/activate   # macOS/Linux
  # or on Windows:  .venv\Scripts\activate
  tunebench train --model ...
  ```
- **Or skip activation** and use `uv run` (uv uses the project’s venv automatically):
  ```bash
  uv run tunebench train --model ...
  uv run python -c "import tunebench; ..."
  ```

**Option B — Create the venv only, then install**

```bash
uv venv                    # create .venv
source .venv/bin/activate  # activate (Windows: .venv\Scripts\activate)
uv sync                    # install dependencies and editable package
```

**Update the lockfile** after changing dependencies in `pyproject.toml`:

```bash
uv lock
```

### Development

Install dev dependencies (pytest, ruff, pyright) with `uv sync`, then:

```bash
make format   # ruff format
make lint     # ruff check
make check    # pyright
make test     # pytest
make all      # format, lint, check, test
```

## Usage

Full fine-tuning:

```bash
tunebench train --model distilgpt2 --dataset data/sample_instructions.json --method full --epochs 3
```

Use `tinyllama` or `mistral` for other models. You can run **LoRA** (`--method lora`, with `--lora-rank`) or **freeze** embeddings/layers (`--freeze-embeddings`, `--freeze-first-n-layers`) to reduce trainable parameters and memory. A sample instruction dataset is in `data/demo.json` (JSON with `instruction` and `output` keys). With ≥10 examples, 20% is used for validation and perplexity is logged; otherwise only training loss is logged.

- **`tunebench weight-diff --model <name> --finetuned <checkpoint_dir>`** — Compare original vs fine-tuned weights per layer (‖W_orig − W_ft‖).
- **`tunebench forgetting-test --model <name> --train-dataset <narrow.json> --eval-dataset <generic.json>`** — Catastrophic forgetting check: eval base → fine-tune → re-eval; report degradation.
- **`tunebench rank-sweep`** / **`tunebench lr-sweep`** — Sweep LoRA ranks or learning rates; results in `results.json`.
- **`tunebench head-importance --model <name>`** — Per-layer, per-head attention importance (weight norm).
- **`tunebench train ... --track-gradient-norm`** — Log gradient L2 norm each step.
- **`tunebench train ... --layer-wise-lr-decay 0.9`** — Layer-wise learning rate decay (earlier layers get smaller LR).

When **CUDA is available**, models load on GPU with bfloat16/float16 by default. See **[docs/](docs/)** for the full command reference and [advanced features](docs/advanced.md).

## Example: full flow

End-to-end demo using the included `data/demo.json`, `data/generic.json`, and `data/narrow.json`. You can run the commands below as-is.

**1. Full fine-tune and save a checkpoint**

```bash
tunebench train \
  --model distilgpt2 \
  --dataset data/demo.json \
  --method full \
  --epochs 2 \
  --output-dir runs/demo-full
```

You’ll see parameter counts, training loss, and (if ≥10 examples) validation loss and perplexity. A checkpoint is written under `runs/demo-full/checkpoint-*`.

**2. See which layers changed (weight drift)**

```bash
tunebench weight-diff \
  --model distilgpt2 \
  --finetuned runs/demo-full/checkpoint-2
```

(Use the actual checkpoint folder name, e.g. `checkpoint-2` or `checkpoint-4` depending on steps.) Output is a table: `Layer 0: 0.002`, `Layer 1: 0.008`, … — larger values mean more change in that layer.

**3. (Optional) Catastrophic forgetting check**

```bash
tunebench forgetting-test \
  --model distilgpt2 \
  --train-dataset data/narrow.json \
  --eval-dataset data/generic.json \
  --epochs 2 \
  --output-dir runs/demo-forgetting
```

The CLI evaluates the base model on `generic.json`, fine-tunes on `narrow.json`, then re-evaluates on `generic.json`. A positive **loss_degradation** means the model got worse on the generic prompts (forgetting).

**4. (Optional) Compare with LoRA**

```bash
tunebench train \
  --model distilgpt2 \
  --dataset data/demo.json \
  --method lora \
  --lora-rank 4 \
  --epochs 2 \
  --output-dir runs/demo-lora
```

Check the logged **trainable_pct** — it should be well under 1%. Then run `weight-diff` on the LoRA checkpoint; base weights drift ~0 because only adapters were trained.

**5. (Optional) Attention head importance**

See which attention heads have the largest weight norms per layer (before or after fine-tuning):

```bash
tunebench head-importance --model distilgpt2
```

**6. (Optional) LoRA rank sweep**

Train with several LoRA ranks (e.g. 2, 4, 8) and compare eval loss and trainable params; results go to `runs/demo-rank-sweep/results.json`:

```bash
tunebench rank-sweep \
  --model distilgpt2 \
  --dataset data/demo.json \
  --ranks 2 4 8 \
  --output-dir runs/demo-rank-sweep \
  --epochs 2
```

**7. (Optional) Learning rate sweep**

Compare a few learning rates; results in `runs/demo-lr-sweep/results.json`:

```bash
tunebench lr-sweep \
  --model distilgpt2 \
  --dataset data/demo.json \
  --learning-rates 1e-5 5e-5 1e-4 \
  --output-dir runs/demo-lr-sweep \
  --epochs 2
```

**8. (Optional) Train with gradient norm and layer-wise LR decay**

Log gradient L2 norm each step and use smaller learning rates for earlier layers:

```bash
tunebench train \
  --model distilgpt2 \
  --dataset data/demo.json \
  --epochs 2 \
  --output-dir runs/demo-advanced \
  --track-gradient-norm \
  --layer-wise-lr-decay 0.9
```

Check the logs for **grad_norm**; with layer-wise decay, earlier layers update more slowly.

---

## Layout

All code lives under **`src/`**; packages are siblings at the same level:

- **`src/tunebench/`** — CLI entry point and shared utils:
  - **`cli.py`** — commands: train, weight-diff, forgetting-test.
  - **`utils.py`** — shared helpers (layer naming, pad token, parameter counts).
- **`src/data/`** — dataset loading, instruction formatting, tokenization.
- **`src/models/`** — model loader, LoRA, layer freezing.
- **`src/training/`** — Trainer integration, callbacks, evaluation.
- **`src/analysis/`** — metrics (perplexity), weight drift, formatting.

Project-level folders (at repo root):

- **`data/`** — custom datasets (e.g. instruction JSON for fine-tuning).
- **`logs/`** — training logs (loss, metrics, etc.).
- **`runs/`** — checkpoints and run outputs.
