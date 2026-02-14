# alphazero-board-games

A lightweight [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) implementation for board games using Monte Carlo Tree Search (MCTS) and a residual policy/value network.  
This repository currently provides three presets:

- `gomoku_9_9` — faster 9×9 setup for iteration
- `gomoku_15_15` — standard 15×15 setup
- `connect4` — classic 6×7 Connect Four with gravity

## Project Structure

```text
alphazero/            Core AlphaZero components (game API, MCTS, network, RL loop)
gomoku_9_9/           9×9 Gomoku preset (config + trainer)
gomoku_15_15/         15×15 Gomoku preset (config + trainer)
connect4/             Connect Four preset (config + trainer)
test/                 Unit tests
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

Install dependencies:

```sh
uv sync
```

## Running Tests

```sh
uv run python -m unittest discover -s test -p '*_test.py'
```

## Training

### 9×9 preset

```sh
uv run python -m gomoku_9_9.trainer
```

### 15×15 preset

```sh
uv run python -m gomoku_15_15.trainer
```

### Connect Four preset

```sh
uv run python -m connect4.trainer
```


### Play against trained models (stdio)

9×9:

```sh
uv run python -m gomoku_9_9.stdio_play
```

15×15:

```sh
uv run python -m gomoku_15_15.stdio_play
```

Connect4:

```sh
uv run python -m connect4.stdio_play
```

Optional flags (all commands):

- `--human-color B|W` to choose who moves first
- `--simulation-num N` to speed up/slow down AI thinking
- `--checkpoint-path PATH_PREFIX` to load a different model prefix

Gomoku move input uses column letters and row numbers, e.g. `E5` or `E 5`. Connect4 move input uses a column number, e.g. `4`.

Both trainers expose all config fields as CLI flags, so you can override defaults:

```sh
uv run python -m gomoku_15_15.trainer -simulation_num 1200 -train_interval 20
```
