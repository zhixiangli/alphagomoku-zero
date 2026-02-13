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

Optional flags (both commands):

- `--human-color B|W` to choose who moves first
- `--simulation-num N` to speed up/slow down AI thinking
- `--checkpoint-path PATH_PREFIX` to load a different model prefix

Move input uses column letters and row numbers, e.g. `E5` or `E 5`.

Both trainers expose all config fields as CLI flags, so you can override defaults:

```sh
uv run python -m gomoku_15_15.trainer -simulation_num 1200 -train_interval 20
```

## Key Default Hyperparameters

Defaults differ by preset and are defined in each config module:

| Parameter | `gomoku_9_9` | `gomoku_15_15` | `connect4` |
|---|---:|---:|---:|
| `rows`, `columns` | 9, 9 | 15, 15 | 6, 7 |
| `n_in_row` | 5 | 5 | 4 |
| `simulation_num` | 400 | 450 | 200 |
| `c_puct` | 1.5 | 1.5 | 1.5 |
| `temp_step` | 8 | 8 | 8 |
| `dirichlet_alpha` | 0.3 | 0.05 | 0.3 |
| `dirichlet_epsilon` | 0.25 | 0.10 | 0.25 |
| `batch_size` | 512 | 512 | 128 |
| `epochs` | 10 | 10 | 10 |
| `train_interval` | 10 | 10 | 20 |
| `lr` | 1e-3 | 1e-3 | 1e-3 |
| `conv_filters` | 128 | 128 | 64 |
| `residual_block_num` | 4 | 6 | 4 |
| `max_sample_pool_size` | 100000 | 200000 | 50000 |

## License

[Apache License 2.0](LICENSE)
