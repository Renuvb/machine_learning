import pandas as pd
import numpy as np
import math

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
        self.sigmoid(batch_data[:,:-1]*self.w1)

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

if __name__ == '__main__':
    nn = BasicNN((8,4,1), None, None)
    print(nn.w2)

    read_data_set()