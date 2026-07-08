"""KNN 分类与回归模型。"""

import numpy as np
from util import euclidean_distance


class KNNClassifier:
    """KNN 分类器。

    默认使用欧氏距离，也可以传入自定义距离函数。
    """

    def __init__(self, n_neighbors=3, distance_metric=euclidean_distance):
        """初始化分类器。

        :param n_neighbors: 预测时使用的近邻数量，默认值为 3。
        :param distance_metric: 距离函数，需要接收 (X_train, x)，并返回距离数组。
        """
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None
        self.distance_metric = distance_metric


    def fit(self, x_train, y_train):
        """保存训练数据。

        KNN 是懒惰学习算法，训练阶段只保存样本和标签。

        :param x_train: 训练特征矩阵，形状为 (n_samples, n_features)。
        :param y_train: 训练标签数组，形状为 (n_samples,)。
        """
        # 统一转成 ndarray，便于后续向量化计算。
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)


    def predict(self, x_test):
        """对多个样本进行分类预测。

        :param x_test: 待预测特征矩阵，形状为 (n_samples, n_features)。
        :return: 预测标签数组，形状为 (n_samples,)。
        """
        return np.array([self._predict_one(x) for x in x_test])

    def _predict_one(self, x):
        """对单个样本进行分类预测。

        :param x: 单个待预测样本，形状为 (n_features,)。
        :return: 预测标签。
        """
        distances = self.distance_metric(self.x_train, x)

        # argsort 返回从近到远的训练样本索引。
        sorted_indics = np.argsort(distances)

        nearest_labels = self.y_train[sorted_indics[: self.k]]
        labels, count = np.unique(nearest_labels, return_counts=True)

        return labels[np.argmax(count)]


class KNNRegression:
    """KNN 回归器。"""

    def __init__(self, n_neighbors=3, distance_matrices=euclidean_distance):
        """初始化回归器。

        :param n_neighbors: 预测时使用的近邻数量，默认值为 3。
        :param distance_matrices: 距离函数，需要接收 (X_train, x)，并返回距离数组。
        """
        self.k = n_neighbors
        self.distance_matrics = distance_matrices
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        """保存训练数据。

        :param x_train: 训练特征矩阵，形状为 (n_samples, n_features)。
        :param y_train: 训练目标值数组，形状为 (n_samples,)。
        """
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """对多个样本进行回归预测。

        :param x_test: 待预测特征矩阵，形状为 (n_samples, n_features)。
        :return: 预测值数组，形状为 (n_samples,)。
        """
        return np.array([self._predict_one(x) for x in x_test])

    def _predict_one(self, x):
        """对单个样本进行回归预测。

        :param x: 单个待预测样本，形状为 (n_features,)。
        :return: 最近 k 个样本目标值的平均数。
        """
        distances = self.distance_matrics(self.x_train, x)

        # argsort 返回从近到远的训练样本索引。
        sorted_indics = np.argsort(distances)

        return np.mean(self.y_train[sorted_indics[: self.k]])
