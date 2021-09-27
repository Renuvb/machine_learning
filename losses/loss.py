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


class CrossEntropy(Loss):
    def loss(self, x, y):
        """
        loss = sum(yi * log(xi))
        :param x: probability [None, k] (0, 1)
        :param y: real label 0 or 1 [None, k]
        :return:
        """
        return np.sum(y * np.log(x)) * -1.0 / x.shape[0]

    def sensitive(self, x, y):
        return y / x * -1.0


if __name__ == '__main__':
    ce = CrossEntropy()
    x = np.array([0.8, 0.1, 0.1]).reshape((1, 3))
    y = np.array([0, 1, 0]).reshape((1,3))
    # y = np.array([1, 0, 0]).reshape((1,3))
    print(ce.loss(x, y))
    print(ce.sensitive(x, y))