#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import itertools
import logging
import random

import numpy
from cachetools import LRUCache

from alphazero.mcts import MCTS


class RL:

    def __init__(self, nnet, env, args):
        self.nnet = nnet
        self.env = env
        self.args = args
        self.sample_pool = LRUCache(maxsize=args.max_sample_pool_size)
        self.nnet.load_weights(self.args.save_weights_path)

    def play_against_itself(self):
        board, player = self.env.get_initial_state()
        boards, players, policies = [], [], []
        mcts = MCTS(self.nnet, self.env, self.args)
        while True:
            actions, pi = mcts.simulate(board, player)
            policy = numpy.zeros(self.args.rows * self.args.columns)
            policy[actions] = pi
            boards.append(board)
            players.append(player)
            policies.append(policy)
            action = numpy.random.choice(actions,
                                         p=0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi))))
            next_board, next_player = self.env.next_state(board, action, player)
            winner = self.env.is_terminal_state(next_board, action, player)
            if winner is not None:
                values = [0 if winner is None else (1 if player == winner else -1) for player in players]
                return [i for i in zip(boards, players, policies, values)]
            board, player = next_board, next_player

    def reinforcement_learning(self):
        for i in itertools.count():
            logging.info("iteration %d:", i)
            samples = self.play_against_itself()
            augmented_data = self.augment_samples(samples)
            self.sample_pool.update([(data[0], data[1:]) for data in augmented_data])
            if self.args.batch_size > len(self.sample_pool):
                logging.info("no enough samples, only %d", len(self.sample_pool))
                continue
            batch = random.sample(self.sample_pool.items(), self.args.batch_size)
            self.nnet.train([(board, player, policy, value) for board, (player, policy, value) in batch])
            if i % self.args.save_weights_interval == 0:
                self.nnet.save_weights(self.args.save_weights_path)

    def augment_samples(self, samples):
        return samples
