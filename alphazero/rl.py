#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import concurrent.futures
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


def _play_game(nnet, game, args):
    """Play a single self-play game and return training samples.

    Each sample is a (canonical_board, policy, value) tuple.
    This is a standalone function so it can be reused by worker processes.
    """
    board, player = game.get_initial_state()
    canonical_boards, players, policies = [], [], []
    mcts = MCTS(nnet, game, args)
    max_moves = args.rows * args.columns
    for i in itertools.count():
        actions, counts = mcts.simulate(board, player)
        pi = counts / numpy.sum(counts)
        policy = numpy.zeros(args.rows * args.columns)
        policy[actions] = pi
        canonical_boards.append(game.get_canonical_form(board, player))
        players.append(player)
        policies.append(policy)

        proba = 0.75 * pi + 0.25 * numpy.random.dirichlet(0.3 * numpy.ones(len(pi)))
        action = (
            actions[numpy.argmax(proba)]
            if i >= args.temp_step
            else numpy.random.choice(actions, p=proba)
        )

        next_board, next_player = game.next_state(board, action, player)
        winner = game.is_terminal_state(next_board, action, player)
        if winner is not None:
            logging.info("winner: %s", winner)
            values = numpy.array([game.compute_reward(winner, p) for p in players])
            return [i for i in zip(canonical_boards, policies, values)]
        assert i < max_moves, (
            "Game exceeded maximum possible moves (%d). "
            "Terminal state detection may be broken." % max_moves
        )
        board, player = next_board, next_player


def _self_play_worker(game, nnet_class, args, model_state_dict):
    """Worker function for parallel self-play in a subprocess.

    Reconstructs the neural network from the serialized state dict
    and plays one complete self-play game.  The model state dict is
    pickled once per task submission; for very large networks consider
    shared-memory approaches if serialization becomes a bottleneck.
    """
    nnet = nnet_class(game, args)
    nnet.model.load_state_dict(model_state_dict)
    return _play_game(nnet, game, args)


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
        logging.info(
            "samples currsize: %d, maxsize: %d",
            len(self.sample_pool),
            self.sample_pool.maxlen,
        )

    def play_against_itself(self):
        return _play_game(self.nnet, self.game, self.args)

    def start(self):
        nnet_class = type(self.nnet)
        num_workers = min(self.args.games_per_train, os.cpu_count() or 1)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            for i in itertools.count():
                logging.info(
                    "training cycle %d: playing %d games in parallel with %d workers",
                    i,
                    self.args.games_per_train,
                    num_workers,
                )

                model_state = self.nnet.model.state_dict()
                futures = [
                    executor.submit(
                        _self_play_worker,
                        self.game,
                        nnet_class,
                        self.args,
                        model_state,
                    )
                    for _ in range(self.args.games_per_train)
                ]
                for future in concurrent.futures.as_completed(futures):
                    samples = future.result()
                    augmented_data = self.game.augment_samples(samples)
                    self.sample_pool.extend(augmented_data)

                logging.info("current sample pool size: %d", len(self.sample_pool))
                if self.args.batch_size <= len(self.sample_pool):
                    self.nnet.train(
                        random.sample(self.sample_pool, self.args.batch_size)
                    )
                if (i + 1) % self.args.persist_interval == 0:
                    persist_sample_pool_thread = threading.Thread(
                        target=self.persist_sample_pool,
                        args=[copy.deepcopy(self.sample_pool)],
                    )
                    persist_sample_pool_thread.start()
                    self.nnet.save_checkpoint(self.args.save_checkpoint_path)

    def persist_sample_pool(self, samples):
        with self.sample_pool_persistence_lock:
            logging.info("persist sample pool start")
            with open(self.args.sample_pool_file, "wb") as f:
                pickle.dump(samples, f)
            logging.info("persist sample pool done")

    def read_sample_pool(self):
        if not os.path.exists(self.args.sample_pool_file):
            return None
        with open(self.args.sample_pool_file, "rb") as f:
            logging.info("load samples from %s", self.args.sample_pool_file)
            return pickle.load(f)
