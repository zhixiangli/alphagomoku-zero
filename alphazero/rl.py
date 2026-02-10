#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import copy
import itertools
import logging
import multiprocessing
import os
import pickle
import random
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy

from alphazero.mcts import MCTS

# ---------------------------------------------------------------------------
# Worker-process globals (set once per worker by _init_self_play_worker)
# ---------------------------------------------------------------------------
_worker_game = None
_worker_nnet = None
_worker_args = None


def _init_self_play_worker(game_class, nnet_class, model_state_dict, args):
    """Initialise a worker process with its own game and neural network."""
    global _worker_game, _worker_nnet, _worker_args
    numpy.random.seed()  # reseed from OS entropy in each worker process
    _worker_args = args
    _worker_game = game_class(args)
    _worker_nnet = nnet_class(_worker_game, args)
    _worker_nnet.model.load_state_dict(model_state_dict)


def _self_play_game(game_index=None):
    """Run one self-play game inside a worker process."""
    return play_one_game(_worker_game, _worker_nnet, _worker_args)


def play_one_game(game, nnet, args):
    """Play a complete self-play game. Returns a list of (board, policy, value) samples."""
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
            values = numpy.array(
                [game.compute_reward(winner, p) for p in players]
            )
            return list(zip(canonical_boards, policies, values))
        assert i < max_moves, (
            "Game exceeded maximum possible moves (%d). "
            "Terminal state detection may be broken." % max_moves
        )
        board, player = next_board, next_player


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
        return play_one_game(self.game, self.nnet, self.args)

    def start(self):
        num_workers = os.cpu_count() or 1
        for round_num in itertools.count():
            logging.info(
                "round %d: playing %d self-play games with %d workers",
                round_num,
                self.args.games_per_training,
                num_workers,
            )
            model_state = copy.deepcopy(self.nnet.model.state_dict())
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp_context,
                initializer=_init_self_play_worker,
                initargs=(
                    type(self.game),
                    type(self.nnet),
                    model_state,
                    self.args,
                ),
            ) as executor:
                all_game_samples = list(
                    executor.map(
                        _self_play_game,
                        range(self.args.games_per_training),
                    )
                )
            for samples in all_game_samples:
                augmented_data = self.game.augment_samples(samples)
                self.sample_pool.extend(augmented_data)
            logging.info("current sample pool size: %d", len(self.sample_pool))
            if self.args.batch_size <= len(self.sample_pool):
                self.nnet.train(random.sample(self.sample_pool, self.args.batch_size))
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
