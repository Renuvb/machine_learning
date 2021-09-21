# coding: utf-8

# from layer import Layer
from layers.layer import Layer
import numpy as np

class Conv2d(Layer):
    def __init__(self, input_shape, filter_size, filter_number, padding=0, with_bias=False):
        super().__init__(input_shape=input_shape)  # batch * height * width * channel
        self.filter_size = filter_size
        self.filter_number = filter_number
        self.padding = padding
        self.filter_shape = (filter_number, filter_size, filter_size, input_shape[3])
        self.w = np.random.rand(filter_number, filter_size, filter_size, input_shape[3])
        if with_bias:
            self.bias = np.random.rand(filter_number)

    def forward(self, input):
        """
        
        :param input: batch * height * width * channel
        :return: 
        """
