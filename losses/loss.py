# coding: utf-8
import numpy as np


class Loss(object):
    def loss(self, x, y):
        pass

    def sensitive(self, x, y):
        pass


class MSELoss(Loss):
    def loss(self, x, y):
        return np.sum((x-y)**2) / 2.0

    def sensitive(self, x, y):
        return x - y