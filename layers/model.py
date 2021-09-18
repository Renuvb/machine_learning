import numpy as np


class MSELoss(object):
    def loss(self, x, y):
        return np.sum((x-y)**2) / 2.0

    def sensitive(self, x, y):
        return x - y


class Model(object):
    def __init__(self):
        self.layers = []
        self.loss_func = MSELoss()
        self.batch_size = 32

    def fit(self, x, y, epoch=1):
        print("x shape:", x.shape)
        for e_i in range(epoch):
            for s in range(0, x.shape[0], self.batch_size):
                current_x = x[s:s+self.batch_size]
                current_y = y[s:s+self.batch_size]
                # forward
                for layer in self.layers:
                    current_x = layer.do_forward(current_x)
                loss = self.loss_func.loss(current_x, current_y)
                sensitive = self.loss_func.sensitive(current_x, current_y)
                for layer in self.layers[::-1]:
                    sensitive = layer.do_backward(sensitive)
                print("%s, loss: %s" % (s, loss))

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output