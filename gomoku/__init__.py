from gomoku.config import GomokuConfig
from gomoku.game import GomokuGame

from alphazero.nnet import AlphaZeroNNet


def configure_module(module):
    """Register Gomoku game bindings with an AlphaZero DI module."""
    module.register(GomokuGame, AlphaZeroNNet)
    return module
