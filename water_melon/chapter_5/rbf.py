# coding: utf-8

import pandas as pd
import numpy as np

# y = sum(w * e ^ (beta * (x-c)^2))
# loss = 0.5 * (y-real_y)^2
# ay / aw = (y-real_y) * e ^ (beta * (x-c)^2)
# ay / abeta = (y - real_y) * w * e ^ (beta * (x-c)^2) * (x-c)^2

class RBF:
    def __init__(self, x, y, c):
        self.x = x # n * m
        self.y = y # n * 1
        self.eta = 0.01
        self.c = c # k * m

        self.n = self.x.shape[0]
        self.m = self.x.shape[1]
        self.k = self.c.shape[0]

        self.beta = np.random.rand(1, self.k) - 0.5
        self.w = np.random.rand(1, self.k) - 0.5
        # self.w = np.array([ 0.07656717, -0.07846214,  0.18227391,  0.22071547]).reshape(1, self.k)
        # self.beta = np.array([-0.04189231,  1.83532317, -0.8179164,  -0.04656909]).reshape(1, self.k)

    def round(self, input_x, input_y, detail_mode=False):
        n = self.n
        m = self.m
        k = self.k
        # print(input_x.shape)
        # print(n.shape)
        t_x = np.matmul(np.ones((n, k, 1)), input_x.reshape((n, 1, m)))
        t_c = np.vstack((self.c,) * k).reshape((n, k, m))
        x_c2 = np.matmul(((t_x - t_c) ** 2), np.ones((n, m, 1))).reshape(n, k)
        e_up = x_c2 * np.matmul(np.ones((n, 1)), self.beta)
        ebeta = np.exp(e_up)  # n * k
        y = np.matmul(ebeta, self.w.T)
        loss = np.sum((y - input_y.reshape((n, 1))) ** 2) / 2.0

        delta_w = np.matmul((y - input_y.reshape((n, 1))).reshape((1, n)), ebeta) * self.eta * (-1)  # 1 * k
        delta_beta = np.matmul((y - input_y.reshape((n, 1))).reshape((1, n)), (np.ones((n,1)) @ self.w) * ebeta * x_c2) * self.eta * (-1)  # 1 * k
        # print(delta_w, delta_beta)
        self.beta += delta_beta
        self.w += delta_w

        if detail_mode:
            print("t_x", t_x)
            print("t_c", t_c)
            print("x_c2", x_c2)
            print("e_up", e_up)
            print("ebeta", ebeta)
            print("y", y)
            print("real_y", input_y)
            print("delta_w", delta_w)
            print("delta_beta", delta_beta)
        return loss

    def run(self, max_round=1000, detail_mode=False):
        # print("round %s, loss %s" % (i, loss))
        for i in range(max_round):
            loss = self.round(self.x, self.y, detail_mode)
            print("round %s loss %s, beta %s, w %s" % (i, loss, self.beta, self.w))
            # print("round %s loss %s, beta %s, w %s" % (i, loss, self.beta, self.w))



def main():
    np_x = np.array([0,0,0,1,1,0,1,1]).reshape((4,2))
    np_y = np_x[:,:1] ^ np_x[:,1:]
    np_c = np.array([0,0,0,1,1,0,1,1]).reshape((4,2))
    print(np_y)
    nn = RBF(np_x, np_y, np_c)
    nn.run(10000, False)

if __name__ == '__main__':
    main()