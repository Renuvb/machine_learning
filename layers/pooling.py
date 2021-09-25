import numpy as np

from machine_learning.layers.layer import Layer


class Pooling2D(Layer):
    def __init__(self, input_shape, pooling_shape):
        super().__init__(input_shape)
        self.pooling_shape = pooling_shape
        self.output_shape = list(self.input_shape)
        self.output_shape[1] = self.output_shape[1] // pooling_shape[0]
        self.output_shape[2] = self.output_shape[2] // pooling_shape[1]
        self.weight_map = None # same shape as input

    def backward(self, sensitive):
        for b_i in range(self.weight_map.shape[0]):
            for c_i in range(self.weight_map.shape[3]):
                for h_i in range(0, self.weight_map.shape[1]):
                    for w_i in range(0, self.weight_map.shape[2]):
                        self.weight_map[b_i][h_i][w_i][c_i] *= sensitive[b_i][h_i//self.pooling_shape[0]][w_i//self.pooling_shape[1]][c_i]
        return self.weight_map


class MaxPooling2D(Pooling2D):
    def __init__(self, input_shape, pooling_shape):
        super().__init__(input_shape, pooling_shape)

    def forward(self, input):
        self.weight_map = np.zeros(input.shape)
        out_shape = list(self.output_shape)
        out_shape[0] = input.shape[0]
        output = np.zeros(out_shape)
        p_h, p_w = self.pooling_shape
        for b_i in range(input.shape[0]):
            for h_i in range(0, input.shape[1], p_h):
                for w_i in range(0, input.shape[2], p_w):
                    for c_i in range(input.shape[3]):
                        tmp = input[b_i][h_i:h_i+p_h,w_i:w_i+p_w,c_i:c_i+1]
                        max_index = np.argmax(tmp)
                        max_i_h = max_index // p_w
                        max_i_w = max_index % p_w
                        self.weight_map[b_i][h_i+max_i_h][w_i+max_i_w][c_i] = 1.0
                        output[b_i][h_i//p_h][w_i//p_w][c_i] = np.max(tmp)
        return output


class AvgPooling2D(Pooling2D):
    def __init__(self, input_shape, pooling_shape):
        super().__init__(input_shape, pooling_shape)

    def forward(self, input):
        p_h, p_w = self.pooling_shape
        self.weight_map = np.ones(input.shape) / p_h / p_w
        out_shape = list(self.output_shape)
        out_shape[0] = input.shape[0]
        output = np.zeros(out_shape)
        for b_i in range(input.shape[0]):
            for h_i in range(0, input.shape[1], p_h):
                for w_i in range(0, input.shape[2], p_w):
                    for c_i in range(input.shape[3]):
                        tmp = input[b_i][h_i:h_i + p_h, w_i:w_i+p_w, c_i:c_i + 1]
                        output[b_i][h_i // p_h][w_i // p_w][c_i] = np.mean(tmp)
        return output