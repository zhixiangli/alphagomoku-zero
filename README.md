# AlphaZero Board Games 🎮

Train and play strong board-game AIs from your terminal with a clean, minimal AlphaZero implementation.

This repository gives you:
- 🧠 A reusable AlphaZero core (MCTS + residual policy/value network)
- ♟️ Three ready-to-run games: **Gomoku 9×9**, **Gomoku 15×15**, and **Connect4**
- 🚀 **Pretrained checkpoints already included** so you can play immediately over standard input/output (no UI required)

If you want a practical, hackable AlphaZero codebase you can understand in one sitting and start using today, this is it.

---

## Why people like this repo

- **Instantly playable**: run a single command and challenge a trained model in your terminal.
- **Simple architecture**: clear modules for game logic, search, network, and training loop.
- **Research-friendly**: every trainer exposes config fields as CLI flags, so experiments are easy.
- **Fast iteration presets**: 9×9 Gomoku for quick cycles, 15×15 for standard board, Connect4 for a classic benchmark.

---

## Included pretrained models ✅

The repository already ships checkpoints in each game folder:

- `gomoku_9_9/data/model.*.pt`
- `gomoku_15_15/data/model.*.pt`
- `connect4/data/model.*.pt`

By default, the stdio player loads the latest checkpoint from each preset path automatically, so you can play right away.

---

## Project structure

```text
alphazero/            Core AlphaZero components (game API, MCTS, network, RL loop)
gomoku_9_9/           9×9 Gomoku preset (config + trainer + stdio player + checkpoint)
gomoku_15_15/         15×15 Gomoku preset (config + trainer + stdio player + checkpoint)
connect4/             Connect Four preset (config + trainer + stdio player + checkpoint)
alphazero/tests/      Core AlphaZero tests (MCTS, RL loop, trainer contracts)
gomoku_9_9/tests/     Gomoku 9×9 game and integration tests
gomoku_15_15/tests/   Gomoku 15×15 game and integration tests
connect4/tests/       Connect4 game and integration tests
```

---

## Quick start (play in under 2 minutes)

### 1) Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### 2) Install dependencies

```bash
uv sync
```

### 3) Play against the pretrained model (stdio)

#### Gomoku 9×9

```bash
uv run python -m gomoku_9_9.stdio_play
```

#### Gomoku 15×15

```bash
uv run python -m gomoku_15_15.stdio_play
```

#### Connect4

```bash
uv run python -m connect4.stdio_play
```

That’s it — no extra setup, no web app, no external service.

---

## How gameplay works (standard input/output)

When you run a `stdio_play` command:

1. The game board is printed in your terminal.
2. You enter a move from your keyboard.
3. The AI thinks with MCTS + network policy/value guidance.
4. The AI move is printed.
5. Repeat until win/loss/draw.

### Move input format

- **Gomoku**: `E5` or `E 5`
- **Connect4**: column number, e.g. `4`

Useful in-game commands:
- `help` for move tips
- `quit` / `exit` to stop

---

## CLI options for stdio players

All stdio players support:

- `--human-color B|W` to choose side (`B` moves first)
- `--simulation-num N` to speed up or strengthen AI search
- `--checkpoint-path PATH_PREFIX` to load another model prefix

Example:

```bash
uv run python -m connect4.stdio_play --human-color W --simulation-num 400
```

---

## Train your own models

Run the trainer for any preset:

```bash
uv run python -m gomoku_9_9.trainer
uv run python -m gomoku_15_15.trainer
uv run python -m connect4.trainer
```

All trainer config fields are exposed as flags. Example:

```bash
uv run python -m gomoku_15_15.trainer -simulation_num 1200 -train_interval 20
```

Checkpoints are saved as timestamped files (e.g. `model.1771199700735.pt`) and automatically discovered by the stdio runner.

---

## Run tests

This project uses **pytest** with markers to separate fast checks from heavier integration/training checks.

### Run the full suite

```bash
uv run pytest
```

### Fast feedback loop (unit + integration, skip slow)

```bash
uv run pytest -m "not slow"
```

### Only slow tests (training/checkpoint related)

```bash
uv run pytest -m slow
```

### Target a specific game package

```bash
uv run pytest connect4/tests
uv run pytest gomoku_9_9/tests
uv run pytest gomoku_15_15/tests
```

---

## Use cases

- Learn AlphaZero concepts from a compact implementation
- Prototype new board games using the shared framework
- Benchmark search/training settings across game types
- Build terminal-based AI game bots quickly

---

## License

Apache-2.0
