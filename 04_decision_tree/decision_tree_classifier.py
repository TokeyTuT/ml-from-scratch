import numpy as np

from utils import BaseEstimator


class Node:
    def __init__(
        self,
        n_samples,
        left=None,
        right=None,
        feature_index=None,
        threshold=None,
        predict_class=None,
    ):
        self.right = right
        self.left = left
        self.feature_index = feature_index
        self.threshold = threshold
        self.n_samples = n_samples
        self.predict_class = predict_class  # 这里传入的应该是用np.unique处理过后的标签值


class DecisionTreeClassifier(BaseEstimator):
    def __init_(self, max_depth=None, min_sample_leaf=1, min_sample_split=2):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.min_sample_split = min_sample_split

        self._root = None
        self._y = None
        self._y_encoded = None

    def fit(self, X, y):
        self._transform_training_data(X, y)
        self._root = self._build_trees(X, y)

        return self

    def predict(self, X):
        self._transform_test_data(X)

        y_pred = np.array(self._predict_instance(x for x in X))
        return self._y[y_pred]

    # 处理训练集，将输入数据转换为数字
    def _transform_training_data(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._y, self._y_encoded = np.unique(y, return_inverse=True)

        X_transform = np.array(np.unique(x) for x in X)
        return X_transform

    # 处理测试集，将输入数据转换为数字
    def _transform_test_data(self, X):
        X = np.array(X)

        return np.unique(X)

    # 将标签值编号变回原型

    # 计算不纯度函数
    def _impurity_measure(self, y):
        y = np.array(y)
        unique_classes, count = np.unique(y, return_counts=True)
        n_samples = y.size
        p = unique_classes / n_samples
        impurity = np.sum(1 - p**2)

        return impurity

    # 分割数据
    def _split_data(self, X, y):
        if len(y) < 1:
            return None, None

        impurity_base = self._impurity_measure(y)
        if impurity_base == 0:
            return None, None

        best_threshold = None
        best_feature_idx = None
        m, n = X.shape
        for feature_idx in range(n):
            sorted_feature = np.sort(set(X[:, feature_idx]))  # 注意要去重之后排序
            possible_threshold = [
                np.mean([i, j] for i, j in zip(sorted_feature, sorted_feature[1:]))
            ]

            for threshold in possible_threshold:
                left_y = y[X[:, feature_idx] < threshold]
                right_y = y[X[:feature_idx] >= threshold]

                left_y_ratio, right_y_ratio = len(left_y) / m, len(right_y) / m
                # 计算基尼指数
                current_impurity = left_y_ratio * self._impurity_measure(
                    left_y
                ) + right_y_ratio * self._impurity_measure(right_y)

            if current_impurity < impurity_base:
                best_threshold = threshold
                impurity_base = current_impurity
                best_feature_idx = feature_idx

        return best_feature_idx, best_threshold

    # 建树
    def _build_trees(self, X, y, depth=0):
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        n_samples = len(y)
        unique_class = np.unique(y, return_counts=True)
        predict_class = np.argmax(np.bincount(unique_class))  # 多数表决

        feature_idx, threshold = self._split_data(X, y)
        X_left = X[X[:, feature_idx] < threshold]
        X_right = X[X[:, feature_idx] >= threshold]
        y_left = y[X[:, feature_idx] < threshold]
        y_right = y[X[:, feature_idx] >= threshold]

        node = Node(
            n_samples=n_samples,
            feature_index=feature_idx,
            threshold=threshold,
            predict_class=predict_class,
        )
        if n_samples > self.min_sample_split:
            node.left = self._build_trees(X_left, y_left, depth=depth + 1)
            node.right = self._build_trees(X_right, y_right, depth=depth + 1)

        # 叶子节点小鱼最小叶节点节点数
        if node.left is None and node.right is None:
            if node.n_samples < self.min_sample_leaf:
                return None

        return node

    # 预测单个样本
    def _predict_instance(self, x):
        curr = self._root

        while curr.left is not None and curr.right is not None:
            if x[curr.feature_index] < curr.threshold:
                curr = curr.left
            else:
                curr = curr.right

        return curr.predict_class
