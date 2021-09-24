from layer import Layer
from dense import Dense
from reshape import Reshape
from conv2d import Conv2d, conv2d, calc_conv_shape, padding as np_padding
import numpy as np
import datetime
from model import Model


def dense_test():
    # np.random.seed(int(datetime.datetime.now().timestamp()))
    n_w = 10
    output_channel = 3
    real_w = np.random.rand(n_w, output_channel) - 0
    dense_model = Model(batch_size=20)
    dense_model.layers.append(Dense(n_w, True, output_channel=output_channel))

    n_data = 2000
    x = np.random.rand(n_data, n_w)
    # x = np.ones((n_data, n_w))
    # x = np.array([1,1,1,1,2,3]).reshape(2,3)
    y = np.matmul(x, real_w)

    # print(x, real_w, y)


    # dense_model.fit(x[:900], y[:900])
    print(dense_model.layers[0].bias)

    dense_model.fit(x, y, epoch=50)

    print(dense_model.layers[0].bias)


def conv2d_test():
    # no padding, stride=1
    filter_num = 1
    padding = 0
    stride = 2
    filter_size = 3
    input_shape = [1000, 5, 5, 1]
    output_shape = calc_conv_shape(input_shape[1:3], (filter_size, filter_size), padding, stride)
    with_bias = False
    print("output_shape", output_shape)

    w = np.random.rand(filter_num, filter_size, filter_size, input_shape[3]) - 0.5
    x = np.random.rand(input_shape[0], input_shape[1],input_shape[2],input_shape[3]) - 0.5
    # y = conv2d(np_padding(x, padding), w).reshape(input_shape[0], output_shape[0]*output_shape[1]*filter_num)
    y = conv2d(np_padding(x, padding), w)
    w2 = np.random.rand(1, output_shape[0], output_shape[1], 1) - 0.5
    z = conv2d(y, w2)

    print("x", x.shape)
    print("y", y.shape)
    print("w", w.shape)
    print("z", z.shape)

    conv_model = Model()
    conv_layer = Conv2d((None, input_shape[1], input_shape[2], input_shape[3]), filter_size, filter_num, padding, stride, with_bias)
    conv_model.layers.append(conv_layer)
    print("conv1 output", conv_layer.output_shape)
    conv_model.layers.append(Conv2d(conv_layer.output_shape, conv_layer.output_shape[1], 1, 0, 1, False))
    # conv_model.layers.append(Reshape(conv_layer.output_shape, [None, output_shape[0]*output_shape[1]*filter_num]))
    conv_model.fit(x, z, epoch=10)




if __name__ == '__main__':
    conv2d_test()