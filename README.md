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

Full fine-tuning (Week 1):

```bash
tunebench train --model distilgpt2 --dataset data/sample_instructions.json --method full --epochs 3
```

Use `tinyllama` or `mistral` for other models. A sample instruction dataset is in `data/sample_instructions.json` (JSON with `instruction` and `output` keys). With ≥10 examples, 20% is used for validation and perplexity is logged; otherwise only training loss is logged.

## Layout

- **`src/tunebench/`** — the package (dataset, model_loader, trainer, metrics, cli). Code only; no data or run artifacts.
- **`data/`** — custom datasets (e.g. instruction JSON for fine-tuning).
- **`logs/`** — training logs (loss, metrics, etc.).
- **`runs/`** — checkpoints and run outputs.
