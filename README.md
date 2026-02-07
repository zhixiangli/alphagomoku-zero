# alphazero-gomoku

An AlphaZero implementation for Gomoku (Five in a Row) using Monte Carlo Tree Search and deep neural networks.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

Install dependencies:

```sh
uv sync
```

## Running Tests

```sh
uv run python -m unittest discover -s test -p '*_test.py'
```

## Training

```sh
uv run python battle.py
```

## Battle Mode

```sh
uv run python battle.py -is_battle 1
```

The battle agent reads JSON commands from stdin and returns the next move.