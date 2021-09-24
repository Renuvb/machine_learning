# coding: utf-8

# from layer import Layer
from layer import Layer
import numpy as np
import os


def calc_conv_shape(input_shape, filter_shape, padding=0, stride=1):
    return int((input_shape[0] - filter_shape[0] + 2 * padding) / stride + 1), int((input_shape[1] - filter_shape[1] + 2 * padding) / stride + 1)


def conv2d(input, filter, stride=1):
    """
    :param input: batch * height * width * channel
    :param filter: filter_number * filter_size * filter_size * channel
    :param stride:
    :return: batch * output_shape[0] * output_shape[1] * filter_number
    """
    # print(input.shape, filter.shape)
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
    if padding == 0:
        return input
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
        conv_output_shape = calc_conv_shape(input_shape[1:3], (filter_size, filter_size), padding, stride)
        self.output_shape = (None, conv_output_shape[0], conv_output_shape[1], filter_number)
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
            delta_w = input_channel * input_h * input_w * batch   conv   filter_num * out_h * out_w * batch -> input_channel * w_h * w_w * filter_num


        backward:
        input_sensitive = batch * padding_output_h * padding_output_w * filter_num   conv   input_channel * w_h * w_w * filter_num -> batch * input_h * input_w * input_channel
        """
        padding_out = padding(sensitive, self.filter_size-1)
        rotated_w = rotate(self.w)
        sensitive_x = conv2d(padding_out, rotated_w)
        if self.padding > 0:
            sensitive_x = sensitive_x[:,self.padding:-self.padding, self.padding:-self.padding,:]
        # print("padding_out %s, rotated_w %s, sensitive shape: %s, output_sensitive shape: %s" % (padding_out.shape, rotated_w.shape, sensitive.shape, sensitive_x.shape))

        swapped_input = np.swapaxes(padding(self.current_input, self.padding), 0, 3)
        swapped_sensitive = np.swapaxes(sensitive, 0, 3)
        # stride
        delta_w = conv2d(swapped_input, swapped_sensitive)
        delta_w = np.swapaxes(delta_w, 0, 3) / self.current_input.shape[0] / self.w.size * self.eta * -1
        self.w += delta_w

        return sensitive_x


def read_data(file_name):
    def split_line(line):
        line = line.strip()
        if line == "":
            return []
        items = line.split(' ')
        return [int(a) for a in items]

    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path) as in_file:
        line = in_file.readline()
        shape = split_line(line)
        item_num = 1
        for num in shape:
            item_num *= num
        data = []
        while len(data) < item_num:
            data.extend(split_line(in_file.readline()))
    return np.array(data).reshape(shape)


def test():
    # input = np.arange(36).reshape((2, 3, 3, 2))
    # # print(padding(input, 1))
    # filter = np.ones((1,2,2,2))
    # # print(conv2d(input, filter, 1))
    # print(rotate(np.arange(4).reshape(1,2,2,1)))
    input = read_data(os.path.join(os.path.dirname(__file__), "../self_data/cnn_input"))
    if len(input.shape) == 3:
        input = np.swapaxes(input.reshape(input.shape[0], input.shape[1], input.shape[2], 1), 0, 3)
    filter = read_data(os.path.join(os.path.dirname(__file__), "../self_data/cnn_filter_nhwc"))
    print(conv2d(padding(input,1), filter))


if __name__ == '__main__':
    test()
