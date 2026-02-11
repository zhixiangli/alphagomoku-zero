from alphazero.nnet import AlphaZeroNNet
from gomoku_15_15.game import GomokuGame


def configure_module(module):
    """Register Gomoku game bindings with an AlphaZero DI module."""
    module.register(GomokuGame, AlphaZeroNNet)
    return module
