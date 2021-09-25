import numpy as np
from machine_learning import losses


class Model(object):
    def __init__(self, batch_size=32, loss=losses.MSELoss):
        self.layers = []
        self.loss_func = loss()
        print(self.loss_func)
        self.batch_size = batch_size

    def fit(self, x, y, epoch=1):
        print("x shape:", x.shape)
        for e_i in range(epoch):
            for s in range(0, x.shape[0], self.batch_size):
                current_x = x[s:s+self.batch_size]
                current_y = y[s:s+self.batch_size]
                # forward
                for layer in self.layers:
                    current_x = layer.do_forward(current_x)
                # print("output", current_x)
                loss = self.loss_func.loss(current_x, current_y)
                sensitive = self.loss_func.sensitive(current_x, current_y)
                # print("loss_sensitive: ", sensitive)
                for layer in self.layers[::-1]:
                    sensitive = layer.do_backward(sensitive)
                #     print("sensitive: ", sensitive)
                # print("epoch: %s, step: %s, losses: %s" % (e_i, s, loss))

            print("epoch: %s, losses: %s" % (e_i, loss))

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output