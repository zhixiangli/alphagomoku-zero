from gomoku.config import GomokuConfig
from gomoku.game import GomokuGame
from gomoku.nnet import GomokuNNet


def configure_module(module):
    """Register Gomoku game bindings with an AlphaZero DI module."""
    module.register(GomokuGame, GomokuNNet)
    return module
