#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import logging
import os
import time

import numpy
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Input, Activation, Flatten, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


class NNet:

    def train(self, data):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def save_checkpoint(self, filename):
        raise NotImplementedError()

    def load_checkpoint(self, filename):
        raise NotImplementedError()


class AlphaZeroNNet(NNet):
    """Game-agnostic AlphaZero neural network with a ResNet architecture.

    Implements the dual-headed ResNet from the AlphaZero paper:
    - Shared residual tower
    - Policy head (softmax over action space)
    - Value head (scalar tanh)

    Uses game.get_canonical_form() directly for board representation.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.model = self.build()
        # Pre-compile prediction function for fast single-sample inference
        self._predict_fn = tf.function(lambda x: self.model(x, training=False))

    def build(self):
        input_shape = (self.args.rows, self.args.columns, 2)
        action_space_size = self.args.rows * self.args.columns

        def build_residual_block(x):
            block = Conv2D(self.args.conv_filters, self.args.conv_kernel, padding="same", data_format='channels_last',
                           kernel_regularizer=l2(self.args.l2))(x)
            block = BatchNormalization(axis=-1)(block)
            block = Activation('relu')(block)
            block = Conv2D(self.args.conv_filters, self.args.conv_kernel, padding="same", data_format='channels_last',
                           kernel_regularizer=l2(self.args.l2))(block)
            block = BatchNormalization(axis=-1)(block)
            block = Add()([x, block])
            block = Activation('relu')(block)
            return block

        input_layer = Input(shape=input_shape)

        residual = Conv2D(self.args.conv_filters, self.args.conv_kernel, padding="same", data_format='channels_last',
                          kernel_regularizer=l2(self.args.l2))(input_layer)
        residual = BatchNormalization(axis=-1)(residual)
        residual = Activation('relu')(residual)
        for _ in range(self.args.residual_block_num):
            residual = build_residual_block(residual)

        policy = Conv2D(2, (1, 1), padding="same", data_format='channels_last', kernel_regularizer=l2(self.args.l2))(
            residual)
        policy = BatchNormalization(axis=-1)(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(action_space_size, activation="softmax")(policy)

        value = Conv2D(1, (1, 1), padding="same", data_format='channels_last', kernel_regularizer=l2(self.args.l2))(
            residual)
        value = BatchNormalization(axis=-1)(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(256, activation='relu')(value)
        value = Dense(1, activation='tanh')(value)

        model = Model(inputs=input_layer, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=self.args.lr))
        return model

    def train(self, data):
        boards, policies, values = zip(*data)
        states = numpy.array(boards)
        policies = numpy.array(policies)
        values = numpy.array(values)
        self.model.fit(x=states, y=[policies, values], batch_size=self.args.batch_size, epochs=self.args.epochs)

    def predict(self, board):
        states = board[numpy.newaxis, ...]
        policy, value = self._predict_fn(states)
        return numpy.asarray(policy[0]), float(value[0][0])

    def save_checkpoint(self, filename):
        self.model.save_weights("%s.%s.weights.h5" % (filename, int(time.time() * 1000)))

    def load_checkpoint(self, filename):
        files = glob.glob(filename + '*.weights.h5')
        if len(files) < 1:
            return
        latest_file = max(files, key=os.path.getmtime)
        self.model.load_weights(latest_file)
        logging.info("load checkpoint from %s", latest_file)
