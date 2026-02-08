#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import logging
import os
import time

import numpy
from keras.layers import Conv2D, BatchNormalization, Input, Activation, Flatten, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from alphazero.nnet import NNet
from gomoku.game import ChessType

numpy.random.seed(1337)  # for reproducibility


class GomokuNNet(NNet):

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.model = self.build()

    def build(self):

        def build_residual_block(input):
            block = Conv2D(self.args.conv_filters, self.args.conv_kernel, padding="same", data_format='channels_first',
                           kernel_regularizer=l2(self.args.l2))(input)
            block = BatchNormalization(axis=1)(block)
            block = Activation('relu')(block)
            block = Conv2D(self.args.conv_filters, self.args.conv_kernel, padding="same", data_format='channels_first',
                           kernel_regularizer=l2(self.args.l2))(block)
            block = BatchNormalization(axis=1)(block)
            block = Add()([input, block])
            block = Activation('relu')(block)
            return block

        input = Input(shape=(self.args.history_num * 2 + 1, self.args.rows, self.args.columns))

        residual = Conv2D(self.args.conv_filters, self.args.conv_kernel, padding="same", data_format='channels_first',
                          kernel_regularizer=l2(self.args.l2))(input)
        residual = BatchNormalization(axis=1)(residual)
        residual = Activation('relu')(residual)
        for _ in range(self.args.residual_block_num):
            residual = build_residual_block(residual)

        policy = Conv2D(2, (1, 1), padding="same", data_format='channels_first', kernel_regularizer=l2(self.args.l2))(
            residual)
        policy = BatchNormalization(axis=1)(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(self.args.columns * self.args.rows, activation="softmax")(policy)

        value = Conv2D(1, (1, 1), padding="same", data_format='channels_first', kernel_regularizer=l2(self.args.l2))(
            residual)
        value = BatchNormalization(axis=1)(value)
        value = Activation('relu')(value)
        value = Dense(256)(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(1, activation='tanh')(value)

        model = Model(inputs=input, outputs=[policy, value])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=self.args.lr))
        # model.summary()
        return model

    def train(self, data):
        boards, players, policies, values = zip(*data)
        states = numpy.zeros((len(players), self.args.history_num * 2 + 1, self.args.rows, self.args.columns))
        for i in range(len(players)):
            states[i] = self.fit_transform(boards[i], players[i])
        policies = numpy.array(policies)
        values = numpy.array(values)
        self.model.fit(x=states, y=[policies, values], batch_size=self.args.batch_size, epochs=self.args.epochs)

    def predict(self, data):
        board, player = data
        states = numpy.zeros((1, self.args.history_num * 2 + 1, self.args.rows, self.args.columns))
        states[0] = self.fit_transform(board, player)
        policy, value = self.model.predict(states)
        return policy[0], value[0]

    def save_weights(self, filename):
        self.model.save_weights("%s.%d" % (filename, time.time()))

    def load_weights(self, filename):
        files = glob.glob(filename + '*')
        if len(files) < 1:
            return
        latest_file = max(files, key=os.path.getctime)
        self.model.load_weights(latest_file)
        logging.info("load weights from %s", latest_file)

    def fit_transform(self, board, player):
        def transform(board, player):
            f = numpy.zeros((self.args.history_num, self.args.rows, self.args.columns))
            actions = [self.game.dec_action(stone) for stone in board.split(self.game.semicolon) if
                       stone and stone[0] == player]
            for i in range(self.args.history_num):
                for (x, y) in actions[:len(actions) - i]:
                    f[self.args.history_num - i - 1][x][y] = 1
            return f

        feature = numpy.zeros((self.args.history_num * 2 + 1, self.args.rows, self.args.columns))
        if player == ChessType.BLACK:
            feature[-1] = numpy.ones((self.args.rows, self.args.columns))
        new_board = self.player_insensitive_board(board, player)
        feature[:self.args.history_num] = transform(new_board, ChessType.BLACK)
        feature[self.args.history_num:self.args.history_num * 2] = transform(new_board, ChessType.WHITE)
        return feature

    def player_insensitive_board(self, board, player):
        assert player != ChessType.EMPTY
        if player == ChessType.BLACK:
            return board
        return "".join([c if c != ChessType.BLACK and c != ChessType.WHITE else self.game.next_player(c) for c in board])
