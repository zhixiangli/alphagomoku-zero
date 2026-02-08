#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy


class MCTS:

    def __init__(self, nnet, env, args):
        self.nnet = nnet
        self.env = env
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
        self.env.log_status(board, numpy.copy(self.visit_count[board]), numpy.copy(self.available_actions[board]))
        return numpy.copy(self.available_actions[board]), numpy.copy(self.visit_count[board])

    def search(self, board, player):
        if board not in self.prior_probability:  # leaf
            return -self._expand(board, player)
        index = self._select(board)
        action = self.available_actions[board][index]
        next_board, next_player = self.env.next_state(board, action, player)
        if next_board not in self.terminal_state:
            self.terminal_state[next_board] = self.env.is_terminal_state(next_board, action, player)
        if self.terminal_state[next_board] is not None:
            value = 1 if player == self.terminal_state[next_board] else 0
        else:
            value = self.search(next_board, next_player)
        self._backup(board, index, value)
        return -value

    def _select(self, board):
        ucb = self.args.c_puct * self.prior_probability[board] * numpy.sqrt(
            self.total_visit_count[board]) / (1.0 + self.visit_count[board])
        ucb += self.mean_action_value[board]
        return numpy.argmax(ucb)

    def _backup(self, board, index, value):
        self.mean_action_value[board][index] = (self.mean_action_value[board][index] * self.visit_count[board][
            index] + value) / (self.visit_count[board][index] + 1.0)
        self.visit_count[board][index] += 1
        self.total_visit_count[board] += 1

    def _expand(self, board, player):
        proba, value = self.nnet.predict([board, player])
        actions = self.env.available_actions(board)
        self.available_actions[board] = actions
        action_proba = proba[actions]
        prob_sum = numpy.sum(action_proba)
        if prob_sum > 0:
            self.prior_probability[board] = action_proba / prob_sum
        else:
            self.prior_probability[board] = numpy.ones(len(actions)) / len(actions)
        self.total_visit_count[board] = 0
        self.mean_action_value[board] = numpy.zeros(len(actions))
        self.visit_count[board] = numpy.zeros(len(actions))
        return value
