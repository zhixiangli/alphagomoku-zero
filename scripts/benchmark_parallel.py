#!/usr/bin/env python3
"""Benchmark self-play speed with and without ProcessPoolExecutor.

Runs a configurable number of self-play games both sequentially (in the
current process) and in parallel (via ProcessPoolExecutor), then reports
the wall-clock time for each approach.

Usage::

    python -m scripts.benchmark_parallel
    python -m scripts.benchmark_parallel --num-games 10 --num-workers 4
    python -m scripts.benchmark_parallel --board-size 5 --n-in-row 3
"""

import argparse
import copy
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

from alphazero.nnet import AlphaZeroNNet
from alphazero.rl import _init_self_play_worker, _self_play_game, play_one_game
from gomoku.game import GomokuGame


def _make_args(board_size, n_in_row):
    """Build a minimal config namespace for benchmarking."""
    return argparse.Namespace(
        rows=board_size,
        columns=board_size,
        n_in_row=n_in_row,
        conv_filters=16,
        conv_kernel=(3, 3),
        residual_block_num=2,
        simulation_num=50,
        c_puct=1.0,
        temp_step=2,
        batch_size=1024,
        epochs=20,
        max_sample_pool_size=10000,
        games_per_training=20,
        lr=1e-3,
        l2=1e-4,
        save_checkpoint_path=os.path.join(tempfile.gettempdir(), "bench_model"),
        sample_pool_file=os.path.join(tempfile.gettempdir(), "bench_samples.pkl"),
    )


def benchmark_sequential(game, nnet, args, num_games):
    """Run *num_games* self-play games sequentially and return elapsed time."""
    start = time.perf_counter()
    results = [play_one_game(game, nnet, args) for _ in range(num_games)]
    elapsed = time.perf_counter() - start
    return elapsed, results


def benchmark_parallel(game, nnet, args, num_games, num_workers):
    """Run *num_games* self-play games in parallel and return elapsed time."""
    model_state = copy.deepcopy(nnet.model.state_dict())
    mp_context = multiprocessing.get_context("spawn")
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
        initializer=_init_self_play_worker,
        initargs=(type(game), type(nnet), model_state, args),
    ) as executor:
        results = list(executor.map(_self_play_game, range(num_games)))
    elapsed = time.perf_counter() - start
    return elapsed, results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark self-play with and without ProcessPoolExecutor",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=6,
        help="Number of self-play games to run (default: 6)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max((os.cpu_count() or 1) - 1, 1),
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=5,
        help="Board size (rows and columns, default: 5)",
    )
    parser.add_argument(
        "--n-in-row",
        type=int,
        default=3,
        help="Consecutive stones to win (default: 3)",
    )
    parser.add_argument(
        "--simulation-num",
        type=int,
        default=50,
        help="MCTS simulations per move (default: 50)",
    )
    args = parser.parse_args(argv)

    bench_args = _make_args(args.board_size, args.n_in_row)
    bench_args.simulation_num = args.simulation_num

    game = GomokuGame(bench_args)
    nnet = AlphaZeroNNet(game, bench_args)

    print(
        f"Board: {args.board_size}x{args.board_size}, "
        f"n_in_row: {args.n_in_row}, "
        f"simulations: {args.simulation_num}, "
        f"games: {args.num_games}, "
        f"workers: {args.num_workers}"
    )
    print()

    # --- Sequential ---
    seq_time, seq_results = benchmark_sequential(game, nnet, bench_args, args.num_games)
    seq_samples = sum(len(r) for r in seq_results)
    print(
        f"Sequential : {seq_time:8.3f}s  ({seq_samples} samples from {args.num_games} games)"
    )

    # --- Parallel ---
    par_time, par_results = benchmark_parallel(
        game, nnet, bench_args, args.num_games, args.num_workers
    )
    par_samples = sum(len(r) for r in par_results)
    print(
        f"Parallel   : {par_time:8.3f}s  ({par_samples} samples from {args.num_games} games)"
    )

    # --- Summary ---
    print()
    if par_time > 0:
        speedup = seq_time / par_time
        print(f"Speedup    : {speedup:.2f}x")
    else:
        print("Parallel time was too small to compute speedup.")


if __name__ == "__main__":
    main()
