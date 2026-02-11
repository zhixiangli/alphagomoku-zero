# alphazero-board-games

A lightweight [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) implementation for board games using Monte Carlo Tree Search (MCTS) and a residual policy/value network.  
This repository currently provides two Gomoku presets:

- `gomoku_9_9` — faster 9×9 setup for iteration
- `gomoku_15_15` — standard 15×15 setup

## Project Structure

```text
alphazero/            Core AlphaZero components (game API, MCTS, network, RL loop, evaluation)
gomoku_9_9/           9×9 Gomoku preset (config + trainer)
gomoku_15_15/         15×15 Gomoku preset (config + trainer)
battle.py             CLI for training, battle mode, and model-vs-model evaluation
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

Both trainers expose all config fields as CLI flags, so you can override defaults:

```sh
uv run python -m gomoku_15_15.trainer -simulation_num 1200 -train_interval 20
```

## Battle Mode

Run a JSON stdin/stdout battle agent (loads checkpoint from `-save_checkpoint_path`):

```sh
uv run python battle.py -is_battle 1
```

### JSON protocol

Input (one JSON object per line):

```json
{
  "command": "NEXT_BLACK",
  "chessboard": "B[77];W[78]"
}
```

Output:

```json
{
  "rowIndex": 7,
  "columnIndex": 6
}
```

`command` must be `NEXT_BLACK` or `NEXT_WHITE`.

## Model Evaluation

Compare two checkpoints against each other:

```sh
uv run python battle.py \
  -eval 1 \
  -save_checkpoint_path ./gomoku_9_9/data/model \
  -eval_checkpoint_path ./gomoku_9_9/data/model2 \
  -num_eval_games 50
```

Optional overrides for the second agent:

- `-eval_simulation_num`
- `-eval_c_puct`

## Key Default Hyperparameters

Defaults differ by preset and are defined in each config module:

| Parameter | `gomoku_9_9` | `gomoku_15_15` |
|---|---:|---:|
| `rows`, `columns` | 9, 9 | 15, 15 |
| `n_in_row` | 5 | 5 |
| `simulation_num` | 400 | 900 |
| `c_puct` | 1.5 | 1.5 |
| `temp_step` | 8 | 12 |
| `batch_size` | 512 | 512 |
| `epochs` | 10 | 10 |
| `train_interval` | 10 | 10 |
| `lr` | 1e-3 | 1e-3 |
| `conv_filters` | 128 | 128 |
| `residual_block_num` | 4 | 6 |
| `max_sample_pool_size` | 100000 | 200000 |

## License

[Apache License 2.0](LICENSE)
