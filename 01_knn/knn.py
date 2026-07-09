"""KNN 分类与回归模型。"""

import numpy as np
from distance import euclidean_distance
from utils.base_estimator import BaseEstimator


def _validate_n_neighbors(n_neighbors):
    """检查近邻数量是否为正整数。"""
    if (
        isinstance(n_neighbors, bool)
        or not isinstance(n_neighbors, (int, np.integer))
        or n_neighbors <= 0
    ):
        raise ValueError("n_neighbors must be a positive integer.")


def _prepare_training_data(x_train, y_train, n_neighbors):
    """转换并检查训练数据。"""
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if x_train.ndim != 2:
        raise ValueError("x_train must be a 2D array.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array.")
    if x_train.shape[0] == 0:
        raise ValueError("x_train and y_train cannot be empty.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must contain the same number of samples.")
    if n_neighbors > x_train.shape[0]:
        raise ValueError("n_neighbors cannot be greater than the number of training samples.")

    return x_train, y_train


def _check_is_fitted(x_train, y_train):
    """检查模型是否已经保存训练数据。"""
    if x_train is None or y_train is None:
        raise ValueError("Model has not been fitted yet. Call fit before predict.")


class KNNClassifier(BaseEstimator):
    """KNN 分类器。

    默认使用欧氏距离，也可以传入自定义距离函数。
    """

    def __init__(self, n_neighbors=3, distance_metric=euclidean_distance):
        """初始化分类器。

        :param n_neighbors: 预测时使用的近邻数量，默认值为 3。
        :param distance_metric: 距离函数，需要接收 (X_train, x)，并返回距离数组。
        """
        _validate_n_neighbors(n_neighbors)
        self.n_neighbors = n_neighbors
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
        self.x_train, self.y_train = _prepare_training_data(
            x_train,
            y_train,
            self.n_neighbors,
        )
        return self


    def predict(self, x_test):
        """对多个样本进行分类预测。

        :param x_test: 待预测特征矩阵，形状为 (n_samples, n_features)。
        :return: 预测标签数组，形状为 (n_samples,)。
        """
        _check_is_fitted(self.x_train, self.y_train)
        return np.array([self._predict_one(x) for x in x_test])

    def _predict_one(self, x):
        """对单个样本进行分类预测。

        :param x: 单个待预测样本，形状为 (n_features,)。
        :return: 预测标签。
        """
        distances = self.distance_metric(self.x_train, x)

        # argsort 返回从近到远的训练样本索引。
        sorted_indices = np.argsort(distances)

        nearest_labels = self.y_train[sorted_indices[: self.n_neighbors]]
        labels, count = np.unique(nearest_labels, return_counts=True)

        return labels[np.argmax(count)]


class KNNRegression(BaseEstimator):
    """KNN 回归器。"""

    def __init__(self, n_neighbors=3, distance_metric=euclidean_distance):
        """初始化回归器。

        :param n_neighbors: 预测时使用的近邻数量，默认值为 3。
        :param distance_metric: 距离函数，需要接收 (X_train, x)，并返回距离数组。
        """
        _validate_n_neighbors(n_neighbors)
        self.n_neighbors = n_neighbors
        self.k = n_neighbors
        self.distance_metric = distance_metric
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        """保存训练数据。

        :param x_train: 训练特征矩阵，形状为 (n_samples, n_features)。
        :param y_train: 训练目标值数组，形状为 (n_samples,)。
        """
        self.x_train, self.y_train = _prepare_training_data(
            x_train,
            y_train,
            self.n_neighbors,
        )
        return self

    def predict(self, x_test):
        """对多个样本进行回归预测。

        :param x_test: 待预测特征矩阵，形状为 (n_samples, n_features)。
        :return: 预测值数组，形状为 (n_samples,)。
        """
        _check_is_fitted(self.x_train, self.y_train)
        return np.array([self._predict_one(x) for x in x_test])

    def _predict_one(self, x):
        """对单个样本进行回归预测。

        :param x: 单个待预测样本，形状为 (n_features,)。
        :return: 最近 k 个样本目标值的平均数。
        """
        distances = self.distance_metric(self.x_train, x)

        # argsort 返回从近到远的训练样本索引。
        sorted_indices = np.argsort(distances)

        return np.mean(self.y_train[sorted_indices[: self.n_neighbors]])
