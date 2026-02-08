from tictactoe.config import TicTacToeConfig
from tictactoe.game import TicTacToeGame
from tictactoe.nnet import TicTacToeNNet


def configure_module(module):
    """Register Tic-Tac-Toe game bindings with an AlphaZero DI module."""
    module.register(TicTacToeGame, TicTacToeNNet)
    return module
