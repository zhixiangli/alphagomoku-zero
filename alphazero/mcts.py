#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy


class MCTS:

    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args

        self.visit_count = {}  # N(s, a) is the visit count
        self.mean_action_value = {}  # Q(s, a) is the mean action value
        self.prior_probability = {}  # P(s, a) is the prior probability of selecting that edge.

        self.terminal_state = {}
        self.total_visit_count = {}
        self.available_actions = {}

    def simulate(self, board, player):
        for _ in range(self.args.simulation_num):
            self.search(board, player)
        self.game.log_status(board, numpy.copy(self.visit_count[board]), numpy.copy(self.available_actions[board]))
        return numpy.copy(self.available_actions[board]), numpy.copy(self.visit_count[board])

    def search(self, board, player):
        if board not in self.prior_probability:  # leaf
            return -self.__expand(board, player)
        index = self.__select(board)
        action = self.available_actions[board][index]
        next_board, next_player = self.game.next_state(board, action, player)
        if next_board not in self.terminal_state:
            self.terminal_state[next_board] = self.game.is_terminal_state(next_board, action, player)
        if self.terminal_state[next_board] is not None:
            value = 1 if player == self.terminal_state[next_board] else 0
        else:
            value = self.search(next_board, next_player)
        self.__backup(board, index, value)
        return -value

    def __select(self, board):
        u = self.args.c_puct * self.prior_probability[board] * numpy.sqrt(
            self.total_visit_count[board]) / (1.0 + self.visit_count[board])
        values = self.mean_action_value[board] + u
        return int(numpy.argmax(values))

    def __backup(self, board, index, value):
        self.mean_action_value[board][index] = (self.mean_action_value[board][index] * self.visit_count[board][
            index] + value) / (self.visit_count[board][index] + 1.0)
        self.visit_count[board][index] += 1
        self.total_visit_count[board] += 1

    def __expand(self, board, player):
        canonical_board = self.game.get_canonical_form(board, player)
        proba, value = self.nnet.predict(canonical_board)
        actions = self.game.available_actions(board)
        self.available_actions[board] = actions
        self.prior_probability[board] = proba[actions] / numpy.sum(proba[actions])
        self.total_visit_count[board] = 1
        self.mean_action_value[board] = numpy.zeros(len(actions))
        self.visit_count[board] = numpy.zeros(len(actions))
        return value
