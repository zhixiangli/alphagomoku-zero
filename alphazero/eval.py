#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import itertools
import logging

import numpy

from alphazero.mcts import MCTS


class Evaluator:
    """Evaluates two AlphaZero agents by playing them against each other.

    Each agent is defined by a neural network and config. They share the same
    game instance (same rules and board size) but may differ in MCTS parameters
    and neural network weights.

    Usage::

        evaluator = Evaluator(game, nnet1, config1, nnet2, config2)
        results = evaluator.evaluate(num_games=50)
        # results == {'agent1_wins': 30, 'agent2_wins': 15, 'draws': 5}
    """

    def __init__(self, game, nnet1, config1, nnet2, config2):
        self.game = game
        self.nnet1 = nnet1
        self.config1 = config1
        self.nnet2 = nnet2
        self.config2 = config2

    def evaluate(self, num_games=50):
        """Play num_games between the two agents, alternating who goes first.

        Returns:
            A dict with keys 'agent1_wins', 'agent2_wins', 'draws'.
        """
        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for i in range(num_games):
            agent1_is_first = (i % 2 == 0)
            result = self._play_game(agent1_is_first)

            if result == 1:
                agent1_wins += 1
            elif result == -1:
                agent2_wins += 1
            else:
                draws += 1

            logging.info(
                "Game %d/%d: %s | Running: agent1 %d - agent2 %d - draws %d",
                i + 1, num_games,
                "agent1 wins" if result == 1 else ("agent2 wins" if result == -1 else "draw"),
                agent1_wins, agent2_wins, draws,
            )

        results = {
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
        }
        logging.info("Evaluation complete: %s", results)
        return results

    def _play_game(self, agent1_is_first):
        """Play a single game.

        Returns 1 if agent1 wins, -1 if agent2 wins, 0 for draw.
        """
        board, first_player = self.game.get_initial_state()
        second_player = self.game.next_player(first_player)

        if agent1_is_first:
            mcts_first = MCTS(self.nnet1, self.game, self.config1)
            mcts_second = MCTS(self.nnet2, self.game, self.config2)
        else:
            mcts_first = MCTS(self.nnet2, self.game, self.config2)
            mcts_second = MCTS(self.nnet1, self.game, self.config1)

        player_mcts = {first_player: mcts_first, second_player: mcts_second}
        player = first_player

        for _ in itertools.count():
            mcts = player_mcts[player]
            actions, counts = mcts.simulate(board, player)

            action = actions[int(numpy.argmax(counts))]

            next_board, next_player = self.game.next_state(board, action, player)
            winner = self.game.is_terminal_state(next_board, action, player)

            if winner is not None:
                if winner == first_player:
                    return 1 if agent1_is_first else -1
                elif winner == second_player:
                    return -1 if agent1_is_first else 1
                else:
                    return 0

            board, player = next_board, next_player
