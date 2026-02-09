# alphazero-board-games

A general-purpose [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) implementation for board games using Monte Carlo Tree Search (MCTS) and deep residual neural networks. Currently supports **Gomoku** (Five in a Row) and **Tic-Tac-Toe**, with an extensible architecture for adding new games.

## Project Structure

```
alphazero/          Core AlphaZero algorithm (game interface, MCTS, neural network, self-play, evaluation)
gomoku/             Gomoku game implementation (15×15 board, 5-in-a-row)
tictactoe/          Tic-Tac-Toe game implementation (3×3 board, 3-in-a-row)
battle.py           Gomoku battle & evaluation entry point (stdin/stdout JSON protocol)
test/               Unit tests
```

### Core Algorithm (`alphazero/`)

| File | Description |
|------|-------------|
| `game.py` | Abstract `Game` interface that every board game must implement |
| `nnet.py` | Dual-headed residual neural network (policy head + value head) built on Keras |
| `mcts.py` | Monte Carlo Tree Search with UCB-based selection and neural network leaf evaluation |
| `rl.py` | Self-play reinforcement learning loop (generates training data and trains the network) |
| `eval.py` | Two-agent tournament evaluation for comparing model checkpoints |
| `config.py` | Shared hyperparameter configuration dataclass |
| `module.py` | Dependency injection module for wiring game, network, and trainer |
| `trainer.py` | CLI helpers and common training entry-point utilities |

### Supported Games

- **Gomoku** (`gomoku/`) — Configurable board size (default 15×15) and win condition (default 5-in-a-row). Includes D4 symmetry data augmentation (8 transformations).
- **Tic-Tac-Toe** (`tictactoe/`) — A 3×3 specialization of Gomoku (3-in-a-row). Extends the Gomoku game and network classes.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Dependencies

- [TensorFlow](https://www.tensorflow.org/) ≥ 2.16
- [Keras](https://keras.io/) ≥ 3.13.1
- [NumPy](https://numpy.org/) ≥ 1.24

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

### Gomoku

```sh
uv run python -m gomoku.trainer
```

Customize board size and hyperparameters:

```sh
uv run python -m gomoku.trainer -rows 9 -columns 9 -n_in_row 4
```

### Tic-Tac-Toe

```sh
uv run python tictactoe/trainer.py
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-rows` | 15 (Gomoku) / 3 (TTT) | Board rows |
| `-columns` | 15 (Gomoku) / 3 (TTT) | Board columns |
| `-n_in_row` | 5 (Gomoku) / 3 (TTT) | Consecutive stones needed to win |
| `-simulation_num` | 500 (Gomoku) / 200 (TTT) | MCTS simulations per move |
| `-batch_size` | 1024 (Gomoku) / 512 (TTT) | Training batch size |
| `-lr` | 5e-3 | Learning rate |
| `-epochs` | 20 | Training epochs per iteration |
| `-c_puct` | 1.0 | MCTS exploration constant |
| `-save_checkpoint_path` | `./data/<game>/model` | Path to save model checkpoints |
| `-residual_block_num` | 2 | Number of residual blocks in the network |
| `-conv_filters` | 256 (Gomoku) / 64 (TTT) | Convolutional filter count |

## Battle Mode (Gomoku)

Start an interactive battle agent that communicates via JSON over stdin/stdout:

```sh
uv run python battle.py -is_battle 1
```

### JSON Protocol

**Input** (one JSON object per line on stdin):

```json
{
  "command": "NEXT_BLACK",
  "chessboard": "<SGF string>"
}
```

`command` is either `NEXT_BLACK` or `NEXT_WHITE`.

**Output** (one JSON object per line on stdout):

```json
{
  "rowIndex": 7,
  "columnIndex": 7
}
```

## Model Evaluation (Gomoku)

Compare two trained model checkpoints by playing them against each other:

```sh
uv run python battle.py -eval 1 -save_checkpoint_path ./data/gomoku/model -eval_checkpoint_path ./data/gomoku/model2 -num_eval_games 50
```

## License

[Apache License 2.0](LICENSE)