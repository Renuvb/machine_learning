# coding: utf-8

from layer import Layer


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape)
        self.output_shape = output_shape

    def forward(self, input):
        output_shape = list(self.output_shape)
        output_shape[0] = input.shape[0]
        return input.reshape(output_shape)

    def backward(self, sensitive):
        # input_shape = list(self.input_shape)
        # input_shape[0] = sensitive.shape[0]
        return sensitive.reshape(self.current_input.shape)