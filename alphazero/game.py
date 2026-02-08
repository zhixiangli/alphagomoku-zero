#!/usr/bin/python3
#  -*- coding: utf-8 -*-


class Game:

    @property
    def action_space_size(self):
        """Return the total number of possible actions in the game."""
        raise NotImplementedError()

    def next_player(self, player):
        raise NotImplementedError()

    def next_state(self, board, action, player):
        raise NotImplementedError()

    def is_terminal_state(self, board, action, player):
        raise NotImplementedError()

    def get_initial_state(self):
        raise NotImplementedError()

    def available_actions(self, board):
        raise NotImplementedError()

    def log_status(self, board, counts, actions):
        raise NotImplementedError()

    def augment_samples(self, samples):
        return samples

    def get_canonical_form(self, board, player):
        """Returns state from the perspective of the current player.

        The neural net always sees the board as if it is Player 1.
        """
        raise NotImplementedError()

    def compute_reward(self, winner, player):
        raise NotImplementedError()
