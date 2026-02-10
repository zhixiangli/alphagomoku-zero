#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import io
import os
import sys
import unittest

# scripts/ is not an installed package; add repo root so we can import it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.benchmark_parallel import (
    benchmark_parallel,
    benchmark_sequential,
    main,
    _make_args,
)
from alphazero.nnet import AlphaZeroNNet
from gomoku.game import GomokuGame


class TestBenchmarkParallel(unittest.TestCase):
    """Smoke tests for the benchmark_parallel script."""

    def setUp(self):
        self.args = _make_args(board_size=3, n_in_row=2)
        self.args.simulation_num = 10
        self.game = GomokuGame(self.args)
        self.nnet = AlphaZeroNNet(self.game, self.args)

    def test_sequential_returns_results(self):
        elapsed, results = benchmark_sequential(self.game, self.nnet, self.args, 1)
        self.assertGreater(elapsed, 0)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

    def test_parallel_returns_results(self):
        elapsed, results = benchmark_parallel(
            self.game, self.nnet, self.args, 1, num_workers=1
        )
        self.assertGreater(elapsed, 0)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

    def test_main_runs_without_error(self):
        """main() with small args should complete and print output."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            main(
                [
                    "--num-games",
                    "1",
                    "--board-size",
                    "3",
                    "--n-in-row",
                    "2",
                    "--simulation-num",
                    "10",
                    "--num-workers",
                    "1",
                ]
            )
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        self.assertIn("Sequential", output)
        self.assertIn("Parallel", output)
        self.assertIn("Speedup", output)


if __name__ == "__main__":
    unittest.main()
