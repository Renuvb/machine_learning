from layer import Layer
from dense import Dense
import numpy as np
import datetime
from model import Model


def dense_test():
    # np.random.seed(int(datetime.datetime.now().timestamp()))
    n_w = 4
    output_channel = 1
    real_w = np.random.rand(n_w, output_channel) - 0
    dense_model = Model(batch_size=20)
    dense_model.layers.append(Dense(n_w, False, output_channel=output_channel))

    n_data = 2000
    x = np.random.rand(n_data, n_w)
    # x = np.ones((n_data, n_w))
    # x = np.array([1,1,1,1,2,3]).reshape(2,3)
    y = np.matmul(x, real_w)

    # print(x, real_w, y)

    # dense_model.fit(x[:900], y[:900])
    dense_model.fit(x, y, epoch=1)



if __name__ == '__main__':
    dense_test()