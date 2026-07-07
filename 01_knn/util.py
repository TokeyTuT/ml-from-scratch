# 这个类中实现了几个常见的距离函数  

import numpy as np

def euclidean_distance(X_train,x):
    '''
    欧式距离
    公式： 各个维度差的平方和再开根
    适用场景: 连续型变量，最常用的通用距离度量。
    :param X_train: 训练集数据，形状: n * r (n 代表训练集的个数，r 代表训练集的维数)
    :param: x: 测试的单样本
    '''

    # distance = []
    # for t in X_train:
    #     distance.append(np.sqrt(np.sum((t - x)**2)))

    # # 注意这里不要排序，不然标签会被搞乱
    # return distance

    # 一行流        
    return np.sqrt(np.sum((X_train - x) ** 2,axis=1))



def manhattan_distance(X_train,x):
    '''
    曼哈顿距离
    公式：各个维度差的绝对值的和
    适用场景: 棋盘格路径移动、高维稀疏数据，或者当某些特征受异常值影响极大时。
    :param X_train: 训练集数据，形状: n * r (n 代表训练集的个数，r 代表训练集的维数)
    :param: x: 测试的单样本
    '''
    return np.sum(np.abs(X_train - x),axis = 1)

def chebyshev_distance(X_train,x):
    '''
    切比雪夫距离
    公式: 各维度差的绝对值中的最大值 max(|x_i - p_i|)
    适用场景: 国际象棋中王移动到目标点需要的步数。
    :param X_train: 训练集数据，形状: n * r (n 代表训练集的个数，r 代表训练集的维数)
    :param: x: 测试的单样本
    '''
    return np.max(np.abs(X_train- x),axis = 1)




# if __name__ == "__main__":
#     X_train = np.array([[170, 55], [165, 50], [180, 90], [185, 95]])
#     x = np.array([170,50])
#     dis = manhattan_distance(X_train=X_train,x=x)
#     print(dis)



