#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import math

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
        proba = self.visit_count[board] / sum(self.visit_count[board])
        self.env.log_status(board, proba)
        return self.available_actions[board], proba

    def search(self, board, player):
        if board not in self.prior_probability:  # leaf
            return -self.__expand(board, player)
        index = self.__select(board)
        action = self.available_actions[board][index]
        next_board, next_player = self.env.next_state(board, action, player)
        if next_board not in self.terminal_state:
            self.terminal_state[next_board] = self.env.is_terminal_state(next_board, action, player)
        if self.terminal_state[next_board] is not None:
            value = 1 if player == self.terminal_state[next_board] else 0
        else:
            value = self.search(next_board, next_player)
        self.__backup(board, index, value)
        return -value

    def __select(self, board):
        best_value = -math.inf
        best_index = None
        for i in range(len(self.available_actions[board])):
            curr_value = self.args.c_puct * self.prior_probability[board][i] * math.sqrt(
                self.total_visit_count[board]) / (1.0 + self.visit_count[board][i])
            curr_value += self.mean_action_value[board][i]
            if curr_value > best_value:
                best_value, best_index = curr_value, i
        return best_index

    def __backup(self, board, index, value):
        self.mean_action_value[board][index] = (self.mean_action_value[board][index] * self.visit_count[board][
            index] + value) / (self.visit_count[board][index] + 1.0)
        self.visit_count[board][index] += 1
        self.total_visit_count[board] += 1

    def __expand(self, board, player):
        proba, value = self.nnet.predict([board, player])
        actions = self.env.available_actions(board)
        self.available_actions[board] = actions
        self.prior_probability[board] = proba[actions]
        self.prior_probability[board] = self.prior_probability[board] / sum(self.prior_probability[board])
        self.total_visit_count[board] = 1
        self.mean_action_value[board] = numpy.zeros(len(actions))
        self.visit_count[board] = numpy.zeros(len(actions))
        return value
