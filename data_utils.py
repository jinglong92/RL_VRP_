import torch as t
import numpy as np


def data_gen(batch, test2save_times, node_size, times):
    tS = t.rand(batch * test2save_times, node_size, 2)  # 坐标0~1之间
    tD = np.random.randint(1, 10, size=(batch * test2save_times, node_size, 1))  # 所有客户的需求[1, 10]的整数
    tD = t.LongTensor(tD)  # 得到的是LongTensor值
    tD[:, 0, 0] = 0  # 仓库点的需求为0

    S = t.rand(batch * times, node_size, 2)
    D = np.random.randint(1, 10, size=(batch * times, node_size, 1))  # 所有客户的需求
    D = t.LongTensor(D)
    D[:, 0, 0] = 0  # 仓库点的需求为0
    return tS, tD, S, D


# tS, tD, S, D = data_gen(16, 2, 5, 6)
# print(tS)
