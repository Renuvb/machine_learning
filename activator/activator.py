import numpy as np


class Activator(object):
    def forward(self, input):
        return input

    def backward(self, sensitve):
        return sensitve


class SigmoidActivator(Activator):
    def __init__(self):
        self.current_y = None

    def forward(self, input):
        self.current_y = 1.0 / (1.0 + np.exp(input * -1))
        return self.current_y

    def backward(self, sensitve):
        return sensitve * self.current_y * (1.0 - self.current_y)


class Relu(Activator):
    def forward(self, input):
        pass

    def backward(self, sensitve):
        pass