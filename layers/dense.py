from machine_learning.layers.layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_channel, with_bias=True, output_channel=1, activation=None):
        super().__init__(input_shape=(None, input_channel), activation=activation)
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.output_shape = (None, output_channel)
        self.w = np.random.rand(input_channel, output_channel) - 0.5
        self.with_bias = with_bias
        if self.with_bias:
            self.bias = np.random.rand(1, output_channel) # o_c

    def forward(self, input): # batch * input_channel
        """
        :param input: batch * input_channel
        :return: batch * output_channel
        """
        output = np.matmul(input, self.w)
        if self.with_bias:
            output += np.matmul(np.ones((input.shape[0], 1)), self.bias)
        return output

    def backward(self, sensitive):
        # sensitive: n * out_channel
        x_sensitive = np.matmul(sensitive, self.w.T)

        delta_w = np.matmul(self.current_input.T, sensitive) * self.eta * -1 / self.current_input.shape[0]
        # print("curent_input", self.current_input)
        # print("delta_w", delta_w)
        # print("w before", self.w)
        self.w = self.w + delta_w
        # print("w after", self.w)

        if self.with_bias:
            self.bias += np.matmul(np.ones((1, sensitive.shape[0])), sensitive) * self.eta * -1 / self.current_input.shape[0]

        return x_sensitive


