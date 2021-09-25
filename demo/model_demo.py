import numpy as np
import sys, os
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(repo_dir)
sys.path.append(repo_dir)
from machine_learning.layers import Dense, Conv2d, Reshape, MaxPooling2D, AvgPooling2D
from machine_learning.layers.conv2d import conv2d, calc_conv_shape, padding as np_padding
import datetime
from machine_learning.models import Model
from machine_learning.activator.activator import SigmoidActivator


def dense_test():
    # np.random.seed(int(datetime.datetime.now().timestamp()))
    n_w = 10
    output_channel = 3
    real_w = np.random.rand(n_w, output_channel) - 0.5
    dense_model = Model(batch_size=20)
    dense_model.layers.append(Dense(n_w, True, output_channel=output_channel, activation=SigmoidActivator))

    n_data = 2000
    x = np.random.rand(n_data, n_w)
    # x = np.ones((n_data, n_w))
    # x = np.array([1,1,1,1,2,3]).reshape(2,3)
    y = np.matmul(x, real_w)
    y = 1.0 / (1.0 + np.exp(y * -1))

    # print(x, real_w, y)

    # dense_model.fit(x[:900], y[:900])
    # print(dense_model.layers[0].bias)

    dense_model.fit(x, y, epoch=50)

    # print(dense_model.layers[0].bias)


def conv2d_test():
    # no padding, stride=1
    filter_num = 1
    padding = 0
    stride = 1
    filter_size = 3
    input_shape = [1000, 5, 5, 1]
    output_shape = calc_conv_shape(input_shape[1:3], (filter_size, filter_size), padding, stride)
    with_bias = False
    print("output_shape", output_shape)

    w = np.random.rand(filter_num, filter_size, filter_size, input_shape[3]) - 0.5
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) - 0.5
    # y = conv2d(np_padding(x, padding), w).reshape(input_shape[0], output_shape[0]*output_shape[1]*filter_num)
    y = conv2d(np_padding(x, padding), w, stride)

    print("x", x.shape)
    print("y", y.shape)
    print("w", w.shape)


    conv_model = Model()
    conv_layer = Conv2d((None, input_shape[1], input_shape[2], input_shape[3]), filter_size, filter_num, padding, stride, with_bias)
    conv_model.layers.append(conv_layer)
    print("conv1 output", conv_layer.output_shape)

    # conv_model.layers.append(Conv2d(conv_layer.output_shape, conv_layer.output_shape[1], 1, 0, 1, False))
    # w2 = np.random.rand(1, output_shape[0], output_shape[1], 1) - 0.5
    # z = conv2d(y, w2)
    # print("z", z.shape)
    # conv_model.fit(x, z, epoch=10)

    # conv_model.layers.append(Reshape(conv_layer.output_shape, [None, output_shape[0]*output_shape[1]*filter_num]))

    conv_model.fit(x, y, epoch=100)


def pooling_test():
    # no padding, stride=1
    filter_num = 1
    padding = 0
    stride = 1
    filter_size = 3
    input_shape = [1000, 6, 6, 1]
    pooling_shape = (2,2)
    output_shape = calc_conv_shape(input_shape[1:3], (filter_size, filter_size), padding, stride)
    with_bias = False
    print("output_shape", output_shape)

    w = np.random.rand(filter_num, filter_size, filter_size, input_shape[3]) - 0.5
    x = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) - 0.5
    # y = conv2d(np_padding(x, padding), w).reshape(input_shape[0], output_shape[0]*output_shape[1]*filter_num)
    y = conv2d(np_padding(x, padding), w, stride)
    z = np.zeros((y.shape[0], y.shape[1]//pooling_shape[0], y.shape[2]//pooling_shape[1], y.shape[3]))
    for b_i in range(y.shape[0]):
        for c_i in range(y.shape[3]):
            for h_i in range(0,y.shape[1],pooling_shape[0]):
                for w_i in range(0,y.shape[2],pooling_shape[1]):
                    z[b_i][h_i//pooling_shape[0]][w_i//pooling_shape[1]][c_i] = np.max(y[b_i][h_i:h_i+pooling_shape[0],w_i:w_i+pooling_shape[1],c_i:c_i+1])

    print("x", x.shape)
    print("y", y.shape)
    print("w", w.shape)
    print("z", z.shape)

    conv_model = Model()
    conv_layer = Conv2d((None, input_shape[1], input_shape[2], input_shape[3]), filter_size, filter_num, padding,
                        stride, with_bias)
    conv_model.layers.append(conv_layer)
    print("conv1 output", conv_layer.output_shape)
    conv_model.layers.append(MaxPooling2D((None, y.shape[1], y.shape[2], y.shape[3]), pooling_shape))
    conv_model.fit(x, z, epoch=100)



if __name__ == '__main__':
    # pooling_test()
    # conv2d_test()
    dense_test()