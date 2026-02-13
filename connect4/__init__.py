from alphazero.nnet import AlphaZeroNNet
from connect4.game import Connect4Game


def configure_module(module):
    module.register(Connect4Game, AlphaZeroNNet)
    return module
