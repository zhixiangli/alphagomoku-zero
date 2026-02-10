from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    """Base configuration for AlphaZero training, shared across all games.

    Game-specific configurations should subclass this and add their own fields.
    The AlphaZero core (MCTS, RL, NNet) depends only on these common fields.
    """

    # Board dimensions: number of rows and columns on the game board.
    # Together they define the action space size (rows * columns).
    rows: int
    columns: int

    # Number of MCTS simulations to run per move. Higher values give stronger
    # play but take longer. Each simulation traverses the game tree once.
    simulation_num: int = 500

    # Exploration constant in the PUCT formula used during MCTS tree search.
    # Controls the trade-off between exploitation (low values) and exploration
    # (high values) when selecting moves in the search tree.
    c_puct: float = 1.0

    # After this many moves in a self-play game, switch from stochastic
    # (temperature-based) move selection to greedy (argmax). Early stochastic
    # moves encourage exploration; later greedy moves improve play quality.
    temp_step: int = 2

    # Mini-batch size for neural network training. Also used as the minimum
    # number of samples required in the pool before training starts.
    batch_size: int = 1024

    # Number of full passes over the training data per training session.
    epochs: int = 20

    # Maximum number of (board, policy, value) samples kept in the replay
    # buffer. Oldest samples are evicted when the pool exceeds this size.
    max_sample_pool_size: int = 50000

    # Save model checkpoint and persist the sample pool every this many
    # self-play iterations.
    persist_interval: int = 50

    # Train the neural network every this many self-play iterations,
    # provided enough samples have been collected (>= batch_size).
    train_interval: int = 20

    # Learning rate for the Adam optimizer used to train the neural network.
    lr: float = 5e-3

    # L2 weight decay (regularization) coefficient for the Adam optimizer.
    # Helps prevent overfitting by penalizing large network weights.
    l2: float = 1e-4

    # Number of convolutional filters in the shared residual tower of the
    # neural network. More filters increase model capacity and training cost.
    conv_filters: int = 256

    # Kernel size for convolutional layers in the residual tower.
    conv_kernel: tuple = (3, 3)

    # Number of residual blocks in the shared tower of the neural network.
    # Deeper networks can learn more complex patterns at higher compute cost.
    residual_block_num: int = 2

    # File path prefix for saving model checkpoints. A timestamp suffix
    # is appended automatically (e.g. "./data/model.1234567890.pt").
    save_checkpoint_path: str = "./data/model"

    # File path for persisting the sample replay pool to disk (pickle format).
    sample_pool_file: str = "./data/samples.pkl"

    @property
    def action_space_size(self) -> int:
        """Size of the action space for grid-based board games."""
        return self.rows * self.columns
