# AlphaZero Neural Network — Design Document

## Architecture Overview

```
alphazero/
├── nnet.py          # NNet (abstract interface) + AlphaZeroNNet (game-agnostic ResNet)
├── config.py        # AlphaZeroConfig (shared hyperparameters)
├── game.py          # Game (abstract interface)
├── mcts.py          # Monte Carlo Tree Search
└── rl.py            # Self-play reinforcement learning loop

gomoku/
├── nnet.py          # GomokuNNet (game-specific feature extraction only)
├── game.py          # GomokuGame
└── config.py        # GomokuConfig
```

## Design Rationale

### Separation of Concerns

| Layer | Responsibility | File |
|-------|---------------|------|
| `NNet` | Abstract interface (train, predict, save/load checkpoint) | `alphazero/nnet.py` |
| `AlphaZeroNNet` | Game-agnostic ResNet with dual policy/value heads | `alphazero/nnet.py` |
| `GomokuNNet` | Gomoku-specific board encoding (`fit_transform`) | `gomoku/nnet.py` |

**`AlphaZeroNNet`** owns the neural network architecture (residual tower, policy head, value head), model compilation, and checkpoint I/O. It is fully game-agnostic — it only depends on `args.rows`, `args.columns`, and architecture hyperparameters.

**`GomokuNNet`** is a thin subclass that implements `train()` and `predict()` by converting Gomoku's SGF board representation into tensor features, then delegating to the parent's Keras model.

### Checkpoint API

Methods were renamed from `save_weights`/`load_weights` to `save_checkpoint`/`load_checkpoint` to:
- Accurately reflect that the operation persists a training checkpoint (not just raw weights)
- Support future extensions (optimizer state, training step counter, metadata)
- Align with standard ML framework conventions (PyTorch, TF)

## Bug-Fix Summary

| Bug | Root Cause | Impact | Fix |
|-----|-----------|--------|-----|
| **Value head layer order** | `Dense(256)` applied before `Flatten()` — Dense operated per-row on spatial dims instead of on the full flattened feature vector | Wrong value head architecture: `rows × 256` intermediate size instead of `256`; independent Dense per spatial row; excessive parameters | Moved `Flatten()` before `Dense(256)` to match the AlphaZero paper architecture |
| **Global `numpy.random.seed(1337)`** | Module-level seed set at import time | Hidden side effect: corrupts reproducibility for any code importing this module; behavior depends on import order | Removed the global seed. Callers should set seeds explicitly if determinism is needed |
| **Timestamp truncation** | `"%d" % time.time()` truncates float seconds to integer | Multiple saves within the same second overwrite each other | Use millisecond precision: `int(time.time() * 1000)` |
| **`os.path.getctime` on Linux** | `getctime` returns inode change time (not creation time) on Linux | Could load wrong checkpoint if files were touched/moved | Changed to `os.path.getmtime` (modification time) |
| **Missing `.weights.h5` extension** | Keras 3.x requires `.weights.h5` suffix for `save_weights`/`load_weights` | `save_checkpoint` fails on Keras ≥ 3.x | Added `.weights.h5` suffix to checkpoint filenames |
| **Shadowed built-in `input`** | Local variable named `input` shadows Python's built-in | Minor: prevents use of `input()` inside `build()` | Renamed to `input_layer` |

## Extending to a New Game

To add a new game (e.g., Connect Four), create three files:

### 1. Game logic (`connect4/game.py`)
```python
from alphazero.game import Game

class Connect4Game(Game):
    # Implement: next_player, next_state, is_terminal_state,
    #            get_initial_state, available_actions, etc.
```

### 2. Game config (`connect4/config.py`)
```python
from alphazero.config import AlphaZeroConfig
from dataclasses import dataclass

@dataclass
class Connect4Config(AlphaZeroConfig):
    rows: int = 6
    columns: int = 7
    # game-specific params...
```

### 3. Neural network adapter (`connect4/nnet.py`)
```python
from alphazero.nnet import AlphaZeroNNet

class Connect4NNet(AlphaZeroNNet):
    def train(self, data):
        # Convert game states to tensors, call self.model.fit(...)

    def predict(self, data):
        # Convert game state to tensor, call self.model.predict(...)
```

The ResNet architecture, checkpoint I/O, and training loop (`alphazero/rl.py`, `alphazero/mcts.py`) require **zero changes**.
