#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import copy
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

    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args
        self.sample_pool = deque(maxlen=args.max_sample_pool_size)
        self.sample_pool_persistence_lock = threading.Lock()

        persisted_sample_pool = self.read_sample_pool()
        if persisted_sample_pool:
            self.sample_pool.extend(persisted_sample_pool)
        logging.info("samples currsize: %d, maxsize: %d", len(self.sample_pool), self.sample_pool.maxlen)

    def play_against_itself(self):
        board, player = self.game.get_initial_state()
        canonical_boards, players, policies = [], [], []
        mcts = MCTS(self.nnet, self.game, self.args)
        for i in itertools.count():
            actions, counts = mcts.simulate(board, player)
            pi = counts / numpy.sum(counts)
            policy = numpy.zeros(self.game.action_space_size)
            policy[actions] = pi
            canonical_boards.append(self.game.get_canonical_form(board, player))
            players.append(player)
            policies.append(policy)

            proba = 0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi)))
            action = actions[numpy.argmax(proba)] if i >= self.args.temp_step else numpy.random.choice(actions, p=proba)

            next_board, next_player = self.game.next_state(board, action, player)
            winner = self.game.is_terminal_state(next_board, action, player)
            if winner is not None:
                logging.info("winner: %c", winner)
                values = numpy.array([self.game.compute_reward(winner, p) for p in players])
                return [i for i in zip(canonical_boards, policies, values)]
            board, player = next_board, next_player

    def start(self):
        for i in itertools.count():
            logging.info("iteration %d:", i)
            samples = self.play_against_itself()
            augmented_data = self.game.augment_samples(samples)
            self.sample_pool.extend(augmented_data)
            logging.info("current sample pool size: %d", len(self.sample_pool))
            if self.args.batch_size > len(self.sample_pool):
                continue
            self.nnet.train(random.sample(self.sample_pool, self.args.batch_size))
            if (i + 1) % self.args.persist_interval == 0:
                persist_sample_pool_thread = threading.Thread(target=self.persist_sample_pool,
                                                              args=[copy.deepcopy(self.sample_pool)])
                persist_sample_pool_thread.start()
                self.nnet.save_checkpoint(self.args.save_checkpoint_path)

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
