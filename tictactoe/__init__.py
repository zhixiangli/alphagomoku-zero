from alphazero.nnet import AlphaZeroNNet
from tictactoe.game import TicTacToeGame


def configure_module(module):
    """Register Tic-Tac-Toe game bindings with an AlphaZero DI module."""
    module.register(TicTacToeGame, AlphaZeroNNet)
    return module
