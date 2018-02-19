#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import itertools
import logging
import random
from collections import deque

import numpy

from alphazero.mcts import MCTS
from gomoku.env import ChessType


class RL:

    def __init__(self, nnet, env, args):
        self.nnet = nnet
        self.env = env
        self.args = args
        self.nnet.load_weights(self.args.save_weights_path)

        self.sample_pool = self.read_sample_pool()
        if not self.sample_pool:
            self.sample_pool = deque(maxlen=args.max_sample_pool_size)
        logging.info("samples currsize: %d, maxsize: %d", len(self.sample_pool), self.sample_pool.maxlen)

    def play_against_itself(self):
        board, player = self.env.get_initial_state()
        boards, players, policies = [], [], []
        mcts = MCTS(self.nnet, self.env, self.args)
        for i in itertools.count():
            actions, counts = mcts.simulate(board, player)
            pi = counts / sum(counts)
            policy = numpy.zeros(self.args.rows * self.args.columns)
            policy[actions] = pi
            boards.append(board)
            players.append(player)
            policies.append(policy)

            pi = 0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi)))
            action = actions[numpy.argmax(pi)]

            next_board, next_player = self.env.next_state(board, action, player)
            winner = self.env.is_terminal_state(next_board, action, player)
            if winner is not None:
                logging.info("winner: %c", winner)
                values = [0 if winner == ChessType.EMPTY else (1 if player == winner else -1) for player in players]
                return [i for i in zip(boards, players, policies, values)]
            board, player = next_board, next_player

    def reinforcement_learning(self):
        for i in itertools.count():
            logging.info("iteration %d:", i)
            samples = self.play_against_itself()
            augmented_data = self.augment_samples(samples)
            self.sample_pool.extend(augmented_data)
            self.persist_sample_pool(self.sample_pool)
            logging.info("current sample pool size: %d", len(self.sample_pool))
            if self.args.batch_size > len(self.sample_pool):
                continue
            self.nnet.train(random.sample(self.sample_pool, self.args.batch_size))
            if i % self.args.save_weights_interval == 0:
                self.nnet.save_weights(self.args.save_weights_path)

    def augment_samples(self, samples):
        return samples

    def persist_sample_pool(self, pool):
        pass

    def read_sample_pool(self):
        pass
