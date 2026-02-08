#!/usr/bin/python3
#  -*- coding: utf-8 -*-

from alphazero.eval import Evaluator
from alphazero.nnet import AlphaZeroNNet
from alphazero.rl import RL


class AlphaZeroModule:
    """Dependency injection module for AlphaZero game training.

    Binds game implementations to their neural network adapters,
    with AlphaZeroNNet as the automatic default binding. Game logic,
    configuration, and network architecture remain fully decoupled:
    the module acts as the single composition root.

    Usage::

        module = AlphaZeroModule()
        module.register(GomokuGame, GomokuNNet)
        trainer = module.create_trainer(GomokuGame, GomokuConfig())
        trainer.start()
    """

    def __init__(self):
        self._nnet_bindings = {}

    def register(self, game_class, nnet_class):
        """Bind a Game subclass to its NNet implementation."""
        self._nnet_bindings[game_class] = nnet_class
        return self

    def resolve_nnet_class(self, game_class):
        """Resolve the NNet class for a game, defaulting to AlphaZeroNNet."""
        return self._nnet_bindings.get(game_class, AlphaZeroNNet)

    def create_trainer(self, game_class, config):
        """Create a configured RL trainer by injecting game class and config.

        Instantiates the game from the config, automatically resolves the
        appropriate neural network implementation, and assembles the RL
        trainer — all without the caller needing to know which NNet to use.

        Args:
            game_class: A Game subclass to instantiate.
            config: An AlphaZeroConfig (or subclass) instance.

        Returns:
            A configured RL trainer ready to start.
        """
        game = game_class(config)
        nnet_class = self.resolve_nnet_class(game_class)
        nnet = nnet_class(game, config)
        return RL(nnet, game, config)

    def create_evaluator(self, game_class, config1, config2):
        """Create an Evaluator to pit two agents against each other.

        Both configs must share the same board dimensions and game rules.
        They may differ in MCTS parameters, neural-network architecture,
        and checkpoint paths.

        Args:
            game_class: A Game subclass to instantiate.
            config1: Config for agent 1.
            config2: Config for agent 2.

        Returns:
            A configured Evaluator ready to run.
        """
        game = game_class(config1)
        nnet_class = self.resolve_nnet_class(game_class)
        nnet1 = nnet_class(game, config1)
        nnet2 = nnet_class(game, config2)
        return Evaluator(game, nnet1, config1, nnet2, config2)
