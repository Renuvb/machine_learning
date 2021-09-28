# coding: utf-8
import numpy as np
import tensorflow as tf
import sys, os
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(repo_dir)
sys.path.append(repo_dir)
import machine_learning as myml


def cnn_test():
    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("train shape:", x_train.shape, y_train.shape)
    print("test shape:", x_test.shape, y_test.shape)
    y_train_mod = np.zeros((y_train.shape[0], 10))
    for i in range(y_train.shape[0]):
        y_train_mod[i][y_train[i]] = 1.0
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

    model = build_model()
    model.dump_info()
    # model.debug(x_train[:1], y_train_mod[:1])
    model.fit(x_train[:200], y_train_mod[:200], epoch=1)


def build_model():
    model = myml.models.Model(loss=myml.losses.CrossEntropyLoss)
    input_shape = [None, 28, 28, 1]

    conv1 = myml.layers.Conv2d(input_shape, filter_size=5, filter_number=1, padding=2, activation=myml.activator.Relu, name='conv1')
    # print("conv1 output", conv1.out_shape())
    pooling1 = myml.layers.MaxPooling2D(conv1.out_shape(), (4, 4), name='pooling1')
    # print("pooling output", pooling1.out_shape())
    conv2 = myml.layers.Conv2d(pooling1.out_shape(), filter_size=4, filter_number=32, activation=myml.activator.Relu, name='conv2')
    # print("conv2", conv2.out_shape())
    pooling2 = myml.layers.MaxPooling2D(conv2.out_shape(), (4, 4), name='pooling2')
    # print("pooling2",pooling2.out_shape())

    reshape_layer = myml.layers.Reshape(pooling2.out_shape(), [None, 32], name='reshape')
    # print(reshape_layer.out_shape())
    dense_layer = myml.layers.Dense(reshape_layer.out_shape()[1], output_channel=10, activation=myml.activator.SigmoidActivator, name='dense')
    # print(dense_layer.out_shape())

    model.layers = [conv1, pooling1, conv2, pooling2, reshape_layer, dense_layer]
    return model


if __name__ == '__main__':
    cnn_test()
