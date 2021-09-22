# coding: utf-8

# from layer import Layer
from layer import Layer
import numpy as np


def calc_conv_shape(input_shape, filter_shape, padding=0, stride=1):
    return int((input_shape[0] - filter_shape[0] + 2 * padding) / stride + 1), int((input_shape[1] - filter_shape[1] + 2 * padding) / stride + 1)


def conv2d(input, filter, stride=1):
    """
    :param input: batch * height * width * channel
    :param filter: filter_number * filter_size * filter_size * channel
    :param stride:
    :return:
    """
    output_shape = calc_conv_shape(input.shape[1:3], filter.shape[1:3], stride=stride)
    filter_number = filter.shape[0]
    output = np.zeros((input.shape[0], output_shape[0], output_shape[1], filter_number))
    # print(output_shape)
    for b_i in range(input.shape[0]):
        for h_i in range(output_shape[0]):
            for w_i in range(output_shape[1]):
                for f_i in range(filter_number):
                    output[b_i][h_i][w_i][f_i] += np.sum(input[b_i][h_i * stride:h_i*stride+filter.shape[1],w_i*stride:w_i*stride+filter.shape[2],:] * filter[f_i])
    return output


def padding(input, padding):
    """
    :param input: n*height*width*channel
    :param padding: int
    :return:
    """
    input_shape = input.shape
    output = np.zeros((input.shape[0], input_shape[1]+padding*2, input_shape[2]+padding*2, input_shape[3]))
    output[:,padding:padding+input_shape[1],padding:padding+input_shape[2],:] = input
    return output


def rotate(input):
    """
    :param input: batch * height * width * channel
    :return:
    """
    output = np.zeros(input.shape)
    for b_i in range(input.shape[0]):
        for h_i in range(input.shape[1]):
            for w_i in range(input.shape[2]):
                output[b_i][h_i][w_i] = input[b_i][input.shape[1]-1-h_i][input.shape[2]-1-w_i]
    return output


class Conv2d(Layer):
    def __init__(self, input_shape, filter_size, filter_number, padding=0, stride=1, with_bias=False):
        super().__init__(input_shape=input_shape)  # batch * height * width * channel
        self.filter_size = filter_size
        self.filter_number = filter_number
        self.padding = padding
        self.stride = stride
        self.filter_shape = (filter_number, filter_size, filter_size, input_shape[3])
        self.output_shape = calc_conv_shape(input_shape[1:3], filter_size, padding, stride)
        self.w = np.random.rand(filter_number, filter_size, filter_size, input_shape[3])
        if with_bias:
            self.bias = np.random.rand(filter_number)

    def forward(self, input):
        """
        :param input: batch * height * width * channel
        :return: batch * out_height * out_width * filter_num
        """
        padding_input = padding(input, self.padding)
        return conv2d(padding_input, self.w, self.stride)

    def backward(self, sensitive):
        """
        gradient_w = conv(input, sensitive)
        input: batch * input_h * input_w * input_channel
        w: filter_num * f_h * f_w * input_channel
        :param sensitive: batch * out_height * out_width * filter_num
        :return:

        update filter:
        for every batch:
            for every filter:
                delta_wi = 1 * input_h * input_w * input_channel   conv   1 * out_h * out_w * (1->input_channel) -> 1 * w_h * w_w * 1
                         = input_channel * input_h * input_w * 1   conv   1 * out_h * out_w * 1 -> input_channel * w_h * w_w * 1
            for all filter:
                delta_w = input_channel * input_h * input_w * 1   conv   filter_num * out_h * out_w * 1 -> input_channel * w_h * w_w * filter_num
        for all_batch:
            delta_w = input_channel * input_h * input_w * batch   conv   filter_num * out_h * out_w * batch -> input_channel * w_h * w_w * filter_num / batch


        backward:
        input_sensitive = batch * padding_output_h * padding_output_w * filter_num   conv   input_channel * w_h * w_w * filter_num -> batch * input_h * input_w * input_channel
        """
        padding_out = padding(sensitive, self.filter_size+1)
        rotated_w = rotate(self.w)
        sensitive_x = conv2d(padding_out, rotated_w)

        swapped_input = np.swapaxes(self.current_input, 0, 3)
        swapped_sensitive = np.swapaxes(sensitive, 0, 3)
        # stride
        delta_w = conv2d(swapped_input, swapped_sensitive)
        delta_w = np.swapaxes(delta_w, 0, 3) / self.current_input.shape[0] * self.eta
        self.w += delta_w

        return sensitive_x


def test():
    input = np.arange(36).reshape((2, 3, 3, 2))
    # print(padding(input, 1))
    filter = np.ones((1, 2,2,2))
    # print(conv2d(input, filter, 1))
    print(rotate(np.arange(4).reshape(1,2,2,1)))


if __name__ == '__main__':
    test()
