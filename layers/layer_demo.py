from layer import Layer
from dense import Dense
import numpy as np
import datetime
from model import Model

def dense_test():
    np.random.seed(int(datetime.datetime.now().timestamp()))
    n_w = 3
    real_w = np.random.rand(n_w) - 0
    dense_model = Model()
    dense_model.layers.append(Dense(n_w, False, output_channel=1))

    n_data = 200
    x = np.random.rand(n_data, n_w)
    y = np.matmul(x, real_w.reshape(n_w, 1))

    # print(x, real_w, y)

    # dense_model.fit(x[:900], y[:900])
    dense_model.fit(x, y)





if __name__ == '__main__':
    dense_test()