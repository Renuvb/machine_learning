import pandas as pd
import numpy as np
import math, random

class BasicNN:
    # 只有一个隐层
    # 输出是一维
    # args: nn_shape: 从输入到输出的维度，如 （8, 4, 1） (w,h,1)
    def __init__(self, nn_shape, train_set, test_set):
        # input n*(w+1) label 在最后一维
        self.nn_shape = nn_shape
        # 对参数进行初始化，默认全0初始化
        self.w1 = np.zeros(shape=nn_shape[:2])
        self.w2 = np.zeros(shape=nn_shape[1:])
        self.training_set = train_set
        self.test_set = test_set

    def train(self):
        pass

    def forward(self, batch_data):
        np_a1 = self.sigmoid(batch_data[:,:-1]*self.w1)
        y = self.sigmoid(np_a1*self.w2)

    def _round(self, batch_data):


    def loss(self, data):
        return

    def sigmoid(self, data):
        return 1/(1+math.exp(data))


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

if __name__ == '__main__':
    nn = BasicNN((8,4,1), None, None)
    print(nn.w2)

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

    nn = BasicNN((dim-1, 4, 1), train_set, test_set)


