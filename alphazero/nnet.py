#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import logging
import os
import time

import numpy
import torch
import torch.nn as nn
import torch.optim as optim


class NNet:
    def train(self, data):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def save_checkpoint(self, filename):
        raise NotImplementedError()

    def load_checkpoint(self, filename):
        raise NotImplementedError()


class _ResidualBlock(nn.Module):
    """Single residual block: two convolutions with skip connection."""

    def __init__(self, filters, kernel_size):
        super().__init__()
        padding = kernel_size[0] // 2
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class _AlphaZeroModel(nn.Module):
    """PyTorch module implementing the AlphaZero dual-headed ResNet."""

    def __init__(self, rows, columns, conv_filters, conv_kernel, residual_block_num):
        super().__init__()
        action_space_size = rows * columns
        padding = conv_kernel[0] // 2

        # Shared residual tower
        self.conv_init = nn.Conv2d(2, conv_filters, conv_kernel, padding=padding)
        self.bn_init = nn.BatchNorm2d(conv_filters)
        self.res_blocks = nn.Sequential(
            *[
                _ResidualBlock(conv_filters, conv_kernel)
                for _ in range(residual_block_num)
            ]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(conv_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * columns, action_space_size)

        # Value head
        self.value_conv = nn.Conv2d(conv_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * columns, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared tower
        h = torch.relu(self.bn_init(self.conv_init(x)))
        h = self.res_blocks(h)

        # Policy head (returns logits; caller applies softmax)
        p = torch.relu(self.policy_bn(self.policy_conv(h)))
        p = torch.flatten(p, 1)
        p = self.policy_fc(p)

        # Value head
        v = torch.relu(self.value_bn(self.value_conv(h)))
        v = torch.flatten(v, 1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class AlphaZeroNNet(NNet):
    """Game-agnostic AlphaZero neural network with a ResNet architecture.

    Implements the dual-headed ResNet from the AlphaZero paper:
    - Shared residual tower
    - Policy head (softmax over action space)
    - Value head (scalar tanh)

    Uses game.get_canonical_form() directly for board representation.

    Performance optimizations:
    - channels_last memory format for faster CPU convolutions
    - torch.jit.optimize_for_inference fuses BatchNorm into Conv layers
    - Pre-allocated input buffer eliminates per-call tensor allocation
    - torch.from_numpy for zero-copy tensor creation
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.model = _AlphaZeroModel(
            args.rows,
            args.columns,
            args.conv_filters,
            args.conv_kernel,
            args.residual_block_num,
        ).to(memory_format=torch.channels_last)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.l2,
        )
        self._predict_buf = torch.empty(
            1, 2, args.rows, args.columns, memory_format=torch.channels_last
        )
        self._frozen_model = None

    def _build_frozen_model(self):
        """Build a JIT-optimized model for fast inference.

        Traces the model and applies torch.jit.optimize_for_inference which
        fuses BatchNorm into Conv layers and other graph-level optimizations.
        Must be rebuilt after training or loading new weights.
        """
        self.model.eval()
        example = torch.randn(1, 2, self.args.rows, self.args.columns).to(
            memory_format=torch.channels_last
        )
        traced = torch.jit.trace(self.model, example)
        self._frozen_model = torch.jit.optimize_for_inference(traced)

    def train(self, data):
        boards, policies, values = zip(*data)
        # Input boards are NHWC; convert to NCHW channels_last for PyTorch
        states = (
            torch.from_numpy(numpy.array(boards))
            .float()
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )
        target_policies = torch.from_numpy(numpy.array(policies)).float()
        target_values = torch.from_numpy(numpy.array(values)).float().unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(
            states, target_policies, target_values
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True
        )

        self.model.train()
        for _ in range(self.args.epochs):
            for batch_states, batch_policies, batch_values in loader:
                self.optimizer.zero_grad(set_to_none=True)
                policy_logits, pred_values = self.model(batch_states)
                policy_loss = torch.nn.functional.cross_entropy(
                    policy_logits, batch_policies
                )
                value_loss = torch.mean((pred_values - batch_values) ** 2)
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()
        self._frozen_model = None

    def predict(self, board):
        if self._frozen_model is None:
            self._build_frozen_model()
        self._predict_buf[0] = torch.from_numpy(board).permute(2, 0, 1)
        with torch.inference_mode():
            policy_logits, value = self._frozen_model(self._predict_buf)
            policy = torch.softmax(policy_logits, dim=1)
        return policy[0].numpy(), value[0][0].numpy()

    def save_checkpoint(self, filename):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "%s.%s.pt" % (filename, int(time.time() * 1000)),
        )

    def load_checkpoint(self, filename):
        files = glob.glob(filename + "*.pt")
        if len(files) < 1:
            return
        latest_file = max(files, key=os.path.getmtime)
        try:
            checkpoint = torch.load(latest_file, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._frozen_model = None
            logging.info("load checkpoint from %s", latest_file)
        except (RuntimeError, KeyError) as e:
            logging.error("failed to load checkpoint from %s: %s", latest_file, e)
