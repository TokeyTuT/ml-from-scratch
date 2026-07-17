import numpy as np

from utils import BaseEstimator


class Node:
    """树节点"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, predict=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.predict = predict


class DecisionTreeRegression(BaseEstimator):
    """
    平方误差回归决策树
    """

    def __init__(self, max_depth=3):
        self.max_depth = max_depth

        self._root = None

    def fit(self, X, y):
        self._root = self._build_tree(X, y)
        return self

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        current = self._root

        while current.feature is not None:
            if x[current.feature] < current.threshold:
                current = current.left
            else:
                current = current.right

        return current.predict

    def _find_best_feature_split(self, X, y):
        """
        寻找最优特征和最优切分点
        """

        _, n_feature = X.shape

        best_feature = None
        best_threshold = None
        best_error = np.inf

        for feature in range(n_feature):
            X_feature = X[:, feature]
            unique_feature = np.unique(X_feature)

            if unique_feature.size < 2:
                continue

            possible_thresholds = (unique_feature[:-1] + unique_feature[1:]) / 2

            for threshold in possible_thresholds:
                left_mask = X_feature < threshold
                right_mask = ~left_mask
                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])

                left_error = np.sum((y[left_mask] - left_mean) ** 2)
                right_error = np.sum((y[right_mask] - right_mean) ** 2)
                error = left_error + right_error

                if error < best_error:
                    best_error = error
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_error, best_threshold

    def _build_tree(self, X, y, depth=0):
        if len(y) == 0:
            return None

        predict_res = np.mean(y)
        node = Node(predict=predict_res)

        if self.max_depth is not None and depth >= self.max_depth:
            return node

        feature, error, threshold = self._find_best_feature_split(X, y)

        if feature is None:
            return node

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node
