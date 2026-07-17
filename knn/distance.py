"""KNN 中常用的距离函数。"""

import numpy as np


def euclidean_distance(X_train, x):
    """计算欧氏距离。

    公式：各维度差值平方和的平方根。
    适用场景：连续型特征，也是 KNN 中最常用的通用距离度量。

    :param X_train: 训练集特征矩阵，形状为 (n_samples, n_features)。
    :param x: 单个待预测样本，形状为 (n_features,)。
    :return: 每个训练样本到 x 的距离，形状为 (n_samples,)。
    """
    return np.sqrt(np.sum((X_train - x) ** 2, axis=1))


def manhattan_distance(X_train, x):
    """计算曼哈顿距离。

    公式：各维度差值绝对值之和。
    适用场景：网格路径、高维稀疏特征，或希望降低单个维度平方放大影响的场景。

    :param X_train: 训练集特征矩阵，形状为 (n_samples, n_features)。
    :param x: 单个待预测样本，形状为 (n_features,)。
    :return: 每个训练样本到 x 的距离，形状为 (n_samples,)。
    """
    return np.sum(np.abs(X_train - x), axis=1)


def chebyshev_distance(X_train, x):
    """计算切比雪夫距离。

    公式：各维度差值绝对值中的最大值。
    适用场景：每一步可以同时沿多个维度移动的距离建模。

    :param X_train: 训练集特征矩阵，形状为 (n_samples, n_features)。
    :param x: 单个待预测样本，形状为 (n_features,)。
    :return: 每个训练样本到 x 的距离，形状为 (n_samples,)。
    """
    return np.max(np.abs(X_train - x), axis=1)
