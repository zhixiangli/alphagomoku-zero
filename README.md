# alphazero-gomoku

An AlphaZero-style Gomoku AI using Monte Carlo Tree Search (MCTS) and deep neural networks.

## Prerequisites

- Python 3.10 or higher

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package with a Keras backend. Choose **one** of the following:

   ```bash
   # TensorFlow backend (recommended, original backend)
   pip install -e ".[tensorflow]"

   # JAX backend
   pip install -e ".[jax]"

   # PyTorch backend
   pip install -e ".[torch]"
   ```

3. For development (includes test dependencies):

   ```bash
   pip install -e ".[dev,tensorflow]"
   ```

## Running

### Training

```bash
python battle.py
```

### Battle Mode

```bash
python battle.py -is_battle 1
```

Run `python battle.py --help` for all available options.

## Testing

```bash
python -m pytest
```

## Upgrading Dependencies

Update dependency versions in `pyproject.toml`, then reinstall:

```bash
pip install -e ".[dev,tensorflow]"
```