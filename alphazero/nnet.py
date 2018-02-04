#!/usr/bin/python3
#  -*- coding: utf-8 -*-


class NNet:

    def train(self, data):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def save_weights(self, filename):
        raise NotImplementedError()

    def load_weights(self, filename):
        raise NotImplementedError()
