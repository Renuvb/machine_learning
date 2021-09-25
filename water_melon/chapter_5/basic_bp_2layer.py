# coding: utf-8
import pandas as pd
import numpy as np
import math, random

class BasicNN_2Layer:
    # 只有一个隐层
    # 输出是一维
    # args: nn_shape: 从输入到输出的维度，如 （8, 4, 4, 1） (w,h,1)
    def __init__(self, nn_shape, train_set, test_set):
        # input n*(w+1) label 在最后一维
        self.nn_shape = nn_shape
        # 对参数进行初始化，默认全0初始化
        self.params = {
            'w1': np.random.rand(nn_shape[0], nn_shape[1]) - 0.5,
            'w2': np.random.rand(nn_shape[1], nn_shape[2]) - 0.5,
            'w3': np.random.rand(nn_shape[2], nn_shape[3]) - 0.5,
            'b1': np.random.rand(1, nn_shape[1]) - 0.5,
            'b2': np.random.rand(1, nn_shape[2]) - 0.5,
            'b3': np.random.rand(1, nn_shape[3]) - 0.5,
        }
        self.training_set = train_set
        self.test_set = test_set
        self.eta = 0.01
        print("nn_shape:",nn_shape)
        # print("train_set: %s, test_set: %s, w1 shape: %s, w2: %s" % (train_set.shape, test_set.shape, self.w1.shape, self.w2.shape))
        for k, val in self.params.items():
            print("%s: %s" % (k, val.shape))

    def train(self, max_round=100, detail_mode=False):
        print("train_data", self.training_set.shape)
        last_loss = 9e20

        def round(batch_data):
            w1 = self.params['w1']
            w2 = self.params['w2']
            w3 = self.params['w3']
            b1 = self.params['b1']
            b2 = self.params['b2']
            b3 = self.params['b3']
            n = batch_data.shape[0]
            # forward
            n_ones = np.ones((n, 1))
            a1 = self.sigmoid(
                np.dot(batch_data[:, :-1], w1) + np.dot(n_ones, b1))
            a2 = self.sigmoid(np.dot(a1, w2) + np.dot(n_ones, b2))
            a3 = self.sigmoid(np.dot(a2, w3) + np.dot(n_ones, b3))

            # backward
            g3 = a3 * (1-a3) * (batch_data[:,-1:] - a3)
            g2 = g3 @ w3.T * a2 * (1 - a2)
            g1 = g2 @ w2.T * a1 * (1 - a1)

            delta_w3 = a2.T @ g3 * self.eta
            delta_w2 = a1.T @ g2 * self.eta
            delta_w1 = batch_data[:,:-1].T @ g1 * self.eta
            delta_b3 = n_ones.T @ g3 * self.eta
            delta_b2 = n_ones.T @ g2 * self.eta
            delta_b1 = n_ones.T @ g1 * self.eta

            w1 += delta_w1
            w2 += delta_w2
            w3 += delta_w3
            b1 += delta_b1
            b2 += delta_b2
            b3 += delta_b3


            loss = 0.5 * np.sum((a3 - batch_data[:, -1:]) ** 2)
            if detail_mode:
                print("batch_data",batch_data)
                print("w1",self.w1)
                print("w2",self.w2)
                print("b1",self.b1)
                print("b2",self.b2)
                print("a1",a1)
                print("y",a3)
                print("g2",g2)
                print("delta_w2",delta_w2)
                print("delta_b2",delta_b2)
                print("g1",g1)
                print("delta_w1",delta_w1)
                print("delta_b1",delta_b1)
            return loss


        for i in range(max_round):
            loss = round(self.training_set)
            # print("round:%s, eta: %s, losses: %s" % (i, self.eta, self.losses(self.training_set)))
            print("round:%s, eta: %s, losses: %s" % (i, self.eta, loss))

        for i in range(max_round):
            for j in range(self.training_set.shape[0]):
                round(self.training_set[j:j+1,:])
            print("round:%s, eta: %s, losses: %s" % (i, self.eta, self.loss(self.training_set)))

        # if last_loss < losses:
        #     self.eta *= 0.5
        #     print("round %s, change eta" % i)

    def loss(self, batch_data, print_detail=False):
        w1 = self.params['w1']
        w2 = self.params['w2']
        w3 = self.params['w3']
        b1 = self.params['b1']
        b2 = self.params['b2']
        b3 = self.params['b3']
        n = batch_data.shape[0]
        # forward
        n_ones = np.ones((n, 1))
        a1 = self.sigmoid(
            np.dot(batch_data[:, :-1], w1) + np.dot(n_ones, b1))
        a2 = self.sigmoid(np.dot(a1, w2) + np.dot(n_ones, b2))
        a3 = self.sigmoid(np.dot(a2, w3) + np.dot(n_ones, b3))
        loss = 0.5 * np.sum((a3 - batch_data[:, -1:]) ** 2)
        if print_detail:
            print("y:",a3)
            print("real_y:", batch_data[:,-1:])
            print("diff:", batch_data[:,-1:]-a3)
            # print("w1, %s"%w1)
            # print("w2, %s"%w2)
        return loss


    def sigmoid(self, data):
        return 1 / (1 + np.exp(0-data))

def read_data_set():
    df_data = pd.read_csv("dataset/water_melon3.0.txt")
    df_data.rename(columns={'好瓜':'label','编号':'no','色泽':'color', '根蒂':'root', '敲声':'sound', '纹理':'texture', '脐部':'umbilical', '触感':'touch', '密度':'density', '含糖率':'sugar',}, inplace=True)
    df_data = df_data[['color', 'root', 'sound', 'texture', 'umbilical', 'touch',
       'density', 'sugar', 'label']]
    print(df_data.columns)
    print(df_data.dtypes)
    return df_data

def one_hot(df_data):
    columns = df_data.columns
    non_label_columns = []
    for col in columns:
        if col != 'label':
            non_label_columns.append(col)
    col_value_map = {} # {'col': {"value1": 1}}
    col_to_start_pos = {}
    size = 0
    for col in non_label_columns:
        col_to_start_pos[col] = size
        if df_data[col].dtype == 'float':
            size += 1
        else:
            value_map = {}
            value_list = df_data[col].drop_duplicates().tolist()
            for i, value in enumerate(value_list):
                value_map[value] = i
            col_value_map[col] = value_map
            size += len(value_list)
    # label
    label_value_to_index = {}
    for i, label in enumerate(df_data['label'].drop_duplicates().tolist()):
        label_value_to_index[label] = i
    if len(label_value_to_index) != 2:
        raise("label value number is not 2, %s" % len(label_value_to_index))
    size += 1 # label
    np_instances = np.zeros((len(df_data), size))
    for col in non_label_columns:
        value_map = col_value_map.get(col)
        start_pos = col_to_start_pos[col]
        for i, value in df_data[col].items():
            if value_map is None:  # float value
                pos = start_pos
                real_value = value
            else:
                pos = value_map[value] + start_pos
                real_value = 1
            np_instances[i][pos] = real_value
    for i, value in df_data['label'].items():
        np_instances[i][size-1] = label_value_to_index[value]
    return np_instances


def main():
    # nd_data = np.random.rand(1, 3)
    # nd_data = np.ones((1, 3))
    # for i in range(nd_data.shape[0]):
    #     s = np.sum(nd_data[i][:-1])
    #     nd_data[i][-1] = s
    # train_set = nd_data
    # test_set = nd_data
    # nn = BasicNN((train_set.shape[1] - 1, 2, 1), nd_data, nd_data)

    df_data = read_data_set()
    print(df_data.head(5))
    np_instances = one_hot(df_data)
    instance_num = np_instances.shape[0]
    dim = np_instances.shape[1]
    print(np_instances[:5])
    print(np_instances.shape)

    test_num = int(np_instances.shape[0] / 5)
    test_indexes = random.sample(range(np_instances.shape[0]), test_num)
    print("test indexes is: %s" % test_indexes)

    # train_set = np.zeros((np_instances.shape[0] - test_num,np_instances.shape[1]))
    # test_set = np.zeros((test_num, np_instances.shape[1]))
    train_set = np.ndarray((0,))
    test_set = np.ndarray((0,))
    for i in range(instance_num):
        if i in test_indexes:
            test_set = np.append(test_set, np_instances[i])
        else:
            train_set = np.append(train_set, np_instances[i])
    train_set = train_set.reshape((instance_num-test_num, dim))
    test_set = test_set.reshape((test_num, dim))
    print("train shape: %s, test shape: %s" % (train_set.shape, test_set.shape))
    # nn = BasicNN((train_set.shape[1] - 1, 10, 1), train_set, test_set)
    nn = BasicNN_2Layer((train_set.shape[1] - 1, 4,4, 1), np_instances, test_set)

    # print("test set losses:", nn.losses(test_set))
    nn.train(10000)
    print("test set losses:", nn.loss(test_set, True))

if __name__ == '__main__':
    main()

