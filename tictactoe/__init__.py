from tictactoe.config import TicTacToeConfig
from tictactoe.game import TicTacToeGame

from alphazero.nnet import AlphaZeroNNet


def configure_module(module):
    """Register Tic-Tac-Toe game bindings with an AlphaZero DI module."""
    module.register(TicTacToeGame, AlphaZeroNNet)
    return module
