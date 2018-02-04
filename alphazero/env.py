#!/usr/bin/python3
#  -*- coding: utf-8 -*-


class Env:

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

    def log_status(self, board, proba):
        raise NotImplementedError()
