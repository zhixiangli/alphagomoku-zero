#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import itertools
import logging
import os
import pickle
import random
import threading
from collections import deque

import numpy

from alphazero.mcts import MCTS


class RL:

    def __init__(self, nnet, env, args):
        self.nnet = nnet
        self.env = env
        self.args = args
        self.sample_pool = deque(maxlen=args.max_sample_pool_size)
        self.sample_pool_persistence_lock = threading.Lock()

        persisted_sample_pool = self.read_sample_pool()
        if persisted_sample_pool:
            self.sample_pool.extend(persisted_sample_pool)
        logging.info("samples currsize: %d, maxsize: %d", len(self.sample_pool), self.sample_pool.maxlen)

    def create_mcts(self):
        return MCTS(self.nnet, self.env, self.args)

    def play_against_itself(self):
        board, player = self.env.get_initial_state()
        boards, players, policies = [], [], []
        mcts = self.create_mcts()
        for i in itertools.count():
            actions, counts = mcts.simulate(board, player)
            pi = counts / numpy.sum(counts)
            policy = numpy.zeros(self.args.rows * self.args.columns)
            policy[actions] = pi
            boards.append(board)
            players.append(player)
            policies.append(policy)

            proba = 0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi)))
            proba /= proba.sum()
            action = actions[numpy.argmax(proba)] if i >= self.args.temp_step else numpy.random.choice(actions, p=proba)

            next_board, next_player = self.env.next_state(board, action, player)
            winner = self.env.is_terminal_state(next_board, action, player)
            if winner is not None:
                logging.info("winner: %c", winner)
                player_set = set(players)
                values = [0 if winner not in player_set else (1 if p == winner else -1) for p in players]
                return list(zip(boards, players, policies, values))
            board, player = next_board, next_player

    def start(self, num_iterations=None):
        iterator = range(num_iterations) if num_iterations is not None else itertools.count()
        for i in iterator:
            logging.info("iteration %d:", i)
            samples = self.play_against_itself()
            augmented_data = self.augment_samples(samples)
            self.sample_pool.extend(augmented_data)
            logging.info("current sample pool size: %d", len(self.sample_pool))
            if self.args.batch_size > len(self.sample_pool):
                continue
            self.nnet.train(random.sample(self.sample_pool, self.args.batch_size))
            if (i + 1) % self.args.persist_interval == 0:
                persist_sample_pool_thread = threading.Thread(target=self.persist_sample_pool,
                                                              args=[list(self.sample_pool)])
                persist_sample_pool_thread.start()
                self.nnet.save_weights(self.args.save_weights_path)

    def augment_samples(self, samples):
        return samples

    def persist_sample_pool(self, samples):
        with self.sample_pool_persistence_lock:
            logging.info("persist sample pool start")
            with open(self.args.sample_pool_file, 'wb') as f:
                pickle.dump(samples, f)
            logging.info("persist sample pool done")

    def read_sample_pool(self):
        if not os.path.exists(self.args.sample_pool_file):
            return None
        with open(self.args.sample_pool_file, 'rb') as f:
            logging.info("load samples from %s", self.args.sample_pool_file)
            return pickle.load(f)
