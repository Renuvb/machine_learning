import pandas as pd
import numpy as np
import os, sys


class ReluActivator(object):
    def forward(self, input):
        return input * (input > 0)

    def backward(self, output):
        return 1.0 * (output > 0)


def padding(input, padding_shape):
    if type(padding_shape) is int:
        padding_shape = (padding_shape, padding_shape)
    origin_shape = input.shape
    if len(origin_shape) == 2:
        output = np.zeros([origin_shape[0] + 2 * padding_shape[0], origin_shape[1] + 2 * padding_shape[1]])
        output[origin_shape[0]:origin_shape[0] + origin_shape[0],
        padding_shape[1]:padding_shape[1] + origin_shape[1]] = input
    elif len(origin_shape) == 3:
        output_shape = [origin_shape[0], origin_shape[1], origin_shape[2]]
        output_shape[1] += 2 * padding_shape[0]
        output_shape[2] += 2 * padding_shape[1]
        output = np.zeros(output_shape)
        output[:, padding_shape[0]:padding_shape[0] + origin_shape[1],
        padding_shape[1]:padding_shape[1] + origin_shape[2]] = input
    else:
        raise Exception("should not reach here, input shape %s not valid" % input.shape)
    return output


def calc_conv_output_shape(input_shape, filter_shape, zero_padding, stride):
    return (input_shape - filter_shape + zero_padding * 2) / stride + 1


def convolution(input, filter, stride=1, merge=True):
    """
    calculate convolution
    :param input: ndarray, 3 dimension, input_channel * input_shape
    :param filter: ndarray, 4 dimension, filter_number * input_channel * filter_shape
    :param stride: default to use self.stride
    :return: ndarray, 3 dimension, filter_number * output_shape
    """
    filter_num = filter.shape[0]
    input_channel = filter.shape[1]
    if input_channel != input.shape[0]:
        raise Exception("convolution input channel number %s not equal with filter_channel_num %s" % (input.shape[0], input_channel))
    filter_shape = np.array(filter.shape[2:])
    input_shape = np.array(input.shape[1:])
    output_shape = (input_shape - filter_shape) / stride + 1
    output_shape = output_shape.astype(int)
    print("convolution input shape %s, filter_shape %s, output shape %s" % (input_shape, filter_shape, output_shape))
    # output_shape = (int(output_shape[0]), int(output_shape[1]))
    # print((filter_num, output_shape[0], output_shape[1]))
    if merge:
        output = np.zeros(shape=(filter_num, output_shape[0], output_shape[1]))
    else:
        output = np.zeros(shape=(filter_num, input_channel, output_shape[0], output_shape[1]))
    for filter_i in range(filter_num):
        for channel_i in range(input_channel):
            image_i = input[channel_i]
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    i_pos = i * stride
                    j_pos = j * stride
                    if merge:
                        output[filter_i][i][j] += np.sum(
                            image_i[i_pos:i_pos + filter_shape[0], j_pos:j_pos + filter_shape[1]] * filter[filter_i][channel_i])
                    else:
                        output[filter_i][channel_i][i][j] += np.sum(
                            image_i[i_pos:i_pos + filter_shape[0], j_pos:j_pos + filter_shape[1]] * filter[filter_i][channel_i])
    return output


def rotate(input):
    origin_shape = input.shape
    element_number = input.size
    row_num = origin_shape[-2]
    col_num = origin_shape[-1]
    matrix_size = origin_shape[-2] * origin_shape[-1] * -1
    matrix_num = int(element_number / matrix_size)
    output = input.reshape(matrix_num, row_num, col_num)
    for k in range(matrix_num):
        for i in range(row_num / 2):
            for j in range(col_num / 2):
                tmp = output[k][i][j]
                output[k][i][j] = output[k][row_num - 1 - i][col_num - 1 - i]
                output[k][row_num - 1 - i][col_num - 1 - i] = tmp
    return output.reshape(origin_shape)


class ConvLayer(object):
    def __init__(self, input_shape, input_channel,
                 filter_shape, filter_number,
                 zero_padding=0, stride=1,
                 activator=ReluActivator):
        self.input_shape = input_shape
        self.input_channel = input_channel
        self.filter_shape = filter_shape
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.w = (np.random.rand(filter_number, input_channel, filter_shape[0], filter_shape[1]) - 0.5) * 1e-3
        self.b = (np.random.rand(filter_number) - 0.5) * 1e-3
        self.activator = activator()
        self.cur_input = None

    def forward(self, input):
        """
        forward calculation
        :param input: channel * shape
        :return: filter_number * filter_shape
        """
        self.cur_input = input
        conv_result = convolution(input, self.w, stride=self.stride)
        if conv_result.shape[0] != self.filter_number:
            raise Exception("convolution layer forward result number %d not equal with filter_number %s" % (
            conv_result.shape[0], self.filter_number))
        for i in range(conv_result.shape[0]):
            conv_result[i] += self.b[i]
        return self.activator.forward(conv_result)

    def backward(self, output):
        """
        calculate gradient, and update parameters
        :param output: filter_number * output_shape
        :return: gradient
        """

        # activator
        output = self.activator.backward(output)
        padding_output = padding(output, (self.filter_shape[0] - 1, self.filter_shape[1] - 1))
        # convolution
        # filter filter_number * input_channel * filter_shape
        # input input_channel * input_shape
        # todo: stride
        input_gradient = np.zeros((self.input_channel, self.input_shape[0], self.input_shape[1])) # input_channel * input_shape
        rotated_w = rotate(self.w) # filter_num * input_chan * filter_shape
        input_gradient = convolution(padding_output, np.rollaxis(rotated_w, 1))
        # print(self.cur_input.shape)
        # print("backward output.shape", output.shape)
        # filter_shape = [output.shape[0], self.cur_input.shape[0],output.shape[1],output.shape[2]]
        # cur_output_shape = [output.shape[0], 1, output.shape[1],output.shape[2]]
        # print(filter_shape)
        # print(cur_output_shape)
        # print("cur_input_shape %s, output shape %s, reshape %s, tile %s" % (self.cur_input.shape, output.shape, cur_output_shape, filter_shape))
        # tiled_output = np.tile(output.reshape(cur_output_shape), filter_shape)
        # print("tiled output shape", tiled_output.shape)

        filter_gradient = np.ndarray(shape=(self.filter_number, self.input_channel, self.filter_shape[0], self.filter_shape[1]))
        reshaped_output = output.reshape((self.filter_number, 1, output.shape[1], output.shape[2]))
        for i in range(self.input_channel):
            filter_gradient[:, i, :, :] = convolution(self.cur_input[i:i+1], reshaped_output)
        # filter_gradient = convolution(self.cur_input, tiled_output,merge=False)
        return input_gradient, filter_gradient


def main():
    # ConvLayer((3,4), 3, (2,2), 2)
    pass


def conv_test():
    nd_input = read_data("cnn_input")
    # nd_input = padding(nd_input, 1)
    nd_filter = read_data("cnn_filter")
    cnn_layer = ConvLayer(nd_input.shape[1:],
                          nd_input.shape[0], filter_shape=nd_filter.shape[2:], filter_number=nd_filter.shape[0],
                          zero_padding=0,
                          stride=1)
    cnn_layer.w = nd_filter
    output = cnn_layer.forward(nd_input)
    print("output", output)
    input_gradient, filter_gradient = cnn_layer.backward(output)
    print("input gradient", input_gradient)
    print("filter gradient", filter_gradient)

    # cl = ConvLayer(nd_input.shape[1:], nd_input.shape[0], nd_filter.shape[2:], nd_filter.shape[0], 1)
    # nd_padding_input = padding(nd_input, 1)
    # conv_output = convolution(nd_padding_input, nd_filter, 2)
    # print(conv_output)


def func_learn():
    # nd_a = np.arange(12).reshape((3,4))
    layer = ConvLayer(input_shape=np.array(5, 5),
                      input_channel=3,
                      filter_shape=np.array(3, 3),
                      filter_number=2,
                      zero_padding=1,
                      stride=1)


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


if __name__ == '__main__':
    # main()
    # func_learn()
    conv_test()

    # print(read_data("cnn_input"))
    # read_data("cnn_filter")
