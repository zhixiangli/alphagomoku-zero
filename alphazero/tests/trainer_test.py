#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import argparse
import unittest
from dataclasses import fields
from unittest.mock import patch, MagicMock

from alphazero.trainer import (
    add_config_args,
    build_config_from_args,
    run_training,
    setup_logging,
)
from gomoku_9_9.config import GomokuConfig


class TestAddConfigArgs(unittest.TestCase):
    def test_adds_all_config_fields(self):
        """All GomokuConfig fields are present as CLI arguments."""
        parser = argparse.ArgumentParser()
        add_config_args(parser, GomokuConfig)
        args = parser.parse_args([])
        expected = {f.name for f in fields(GomokuConfig)}
        self.assertEqual(expected, set(vars(args).keys()))

    def test_cli_overrides(self):
        """CLI values override defaults."""
        parser = argparse.ArgumentParser()
        add_config_args(parser, GomokuConfig)
        args = parser.parse_args(["-batch_size", "64", "-lr", "0.001"])
        self.assertEqual(args.batch_size, 64)
        self.assertAlmostEqual(args.lr, 0.001)

    def test_defaults_match_config(self):
        """Argparse defaults match the dataclass defaults."""
        parser = argparse.ArgumentParser()
        add_config_args(parser, GomokuConfig)
        args = parser.parse_args([])
        self.assertEqual(args.save_checkpoint_path, "./gomoku_9_9/data/model")
        self.assertEqual(args.sample_pool_file, "./gomoku_9_9/data/samples.pkl")
        self.assertEqual(args.simulation_num, 400)
        self.assertEqual(args.rows, 9)
        self.assertEqual(args.n_in_row, 5)


class TestBuildConfigFromArgs(unittest.TestCase):
    def test_builds_correct_config(self):
        """build_config_from_args returns a valid config instance."""
        parser = argparse.ArgumentParser()
        add_config_args(parser, GomokuConfig)
        cli_args = parser.parse_args(["-batch_size", "128"])
        config = build_config_from_args(GomokuConfig, cli_args)
        self.assertIsInstance(config, GomokuConfig)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.rows, 9)

    def test_ignores_extra_args(self):
        """Extra CLI flags not in the config are silently ignored."""
        parser = argparse.ArgumentParser()
        parser.add_argument("-logpath", default="./log")
        add_config_args(parser, GomokuConfig)
        cli_args = parser.parse_args([])
        config = build_config_from_args(GomokuConfig, cli_args)
        self.assertIsInstance(config, GomokuConfig)
        self.assertFalse(hasattr(config, "logpath"))


class TestRunTraining(unittest.TestCase):
    def test_calls_module_and_starts_training(self):
        """run_training wires module, loads checkpoint, and starts."""
        mock_module = MagicMock()
        mock_trainer = MagicMock()
        mock_module.create_trainer.return_value = mock_trainer

        config = MagicMock()
        config.save_checkpoint_path = "/tmp/model"

        game_class = type("FakeGame", (), {})

        run_training(mock_module, game_class, config)

        mock_module.create_trainer.assert_called_once_with(game_class, config)
        mock_trainer.nnet.load_checkpoint.assert_called_once_with("/tmp/model")
        mock_trainer.start.assert_called_once()


class TestSetupLogging(unittest.TestCase):
    @patch("alphazero.trainer.logging")
    def test_adds_handlers(self, mock_logging):
        """setup_logging adds file handler only by default."""
        mock_root = MagicMock()
        mock_logging.getLogger.return_value = mock_root
        setup_logging("/tmp/test.log")
        self.assertEqual(mock_root.addHandler.call_count, 1)
        mock_root.setLevel.assert_called_once()

    def test_creates_parent_directory(self):
        """setup_logging creates parent directories if they don't exist."""
        import tempfile
        import os
        import shutil

        tmpdir = tempfile.mkdtemp()
        logpath = os.path.join(tmpdir, "subdir", "nested", "test.log")
        try:
            setup_logging(logpath)
            self.assertTrue(os.path.exists(logpath))
        finally:
            shutil.rmtree(tmpdir)


class TestGomokuTrainerMain(unittest.TestCase):
    def test_main_is_importable(self):
        """gomoku_9_9.trainer.main exists and is callable."""
        from gomoku_9_9.trainer import main

        self.assertTrue(callable(main))

    @patch("gomoku_9_9.trainer.run_training")
    @patch("gomoku_9_9.trainer.setup_logging")
    def test_main_wires_correctly(self, mock_logging, mock_run):
        """main() parses args, builds config, and calls run_training."""
        from gomoku_9_9.trainer import main

        with patch(
            "sys.argv", ["trainer", "-rows", "9", "-columns", "9", "-n_in_row", "3"]
        ):
            main()
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        # args[2] is the config
        config = args[2]
        self.assertEqual(config.rows, 9)
        self.assertEqual(config.columns, 9)
        self.assertEqual(config.n_in_row, 3)
        self.assertEqual(config.save_checkpoint_path, "./gomoku_9_9/data/model")
        self.assertEqual(config.sample_pool_file, "./gomoku_9_9/data/samples.pkl")


if __name__ == "__main__":
    unittest.main()
