.PHONY: format lint check test all

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

check:
	uv run pyright src

test:
	uv run pytest tests -v

all: format lint check test
