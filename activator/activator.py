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
        input = np.amax([input, np.full(input.shape, -14)], axis=0)
        input = np.amin([input, np.full(input.shape, 14)], axis=0)
        self.current_y = 1.0 / (1.0 + np.exp(input * -1))
        return self.current_y

    def backward(self, sensitve):
        return sensitve * self.current_y * (1.0 - self.current_y)


class Relu(Activator):
    def __init__(self):
        self.current_valid_flag = None

    def forward(self, input):
        self.current_valid_flag = (input > 0)
        return input * self.current_valid_flag

    def backward(self, sensitve):
        return sensitve * self.current_valid_flag