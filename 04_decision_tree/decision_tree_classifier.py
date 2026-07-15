import numpy as np

from utils import BaseEstimator,accuracy_score

class Node:
    """决策树中的一个节点。"""

    def __init__(
        self,
        n_samples,
        left=None,
        right=None,
        feature_index=None,
        threshold=None,
        predict_class=None,
    ):
        self.n_samples = n_samples
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        # 节点保存当前样本中的多数类，叶子节点直接使用它进行预测。
        self.predict_class = predict_class


class DecisionTreeClassifier(BaseEstimator):
    """使用 CART 思路实现的分类决策树。"""

    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        self._root = None
        self.classes_ = None
        self.n_features_in_ = None

        self._validate_hyperparameters()

    def fit(self, X, y):
        """根据训练数据递归构建分类决策树。

        :param X: 训练特征矩阵，形状为 (n_samples, n_features)。
        :param y: 训练标签数组，形状为 (n_samples,)。
        :return: 训练后的模型自身。
        """
        self._validate_hyperparameters()
        X, y = self._validate_training_data(X, y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._root = self._build_trees(X, y)

        return self

    def predict(self, X):
        """对多个样本进行分类预测。

        :param X: 待预测特征矩阵，形状为 (n_samples, n_features)。
        :return: 预测标签数组，形状为 (n_samples,)。
        """
        self._check_is_fitted()
        X = self._validate_prediction_data(X)
        return np.array([self._predict_instance(x) for x in X])

    def _impurity_measure(self, y):
        """计算标签数组的 Gini 不纯度。"""
        y = np.asarray(y)
        if y.size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.size
        return float(1 - np.sum(probabilities**2))

    def _split_data(self, X, y):
        """枚举特征和候选阈值，返回加权 Gini 最小的划分。"""
        n_samples, n_features = X.shape
        parent_impurity = self._impurity_measure(y)

        best_impurity = parent_impurity
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):
            feature_values = np.unique(X[:, feature_index])
            if feature_values.size < 2:
                continue

            # 相邻特征值的中点能够保证训练样本被明确分到一侧。
            possible_thresholds = (
                feature_values[:-1] + feature_values[1:]
            ) / 2

            for threshold in possible_thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask
                left_count = np.sum(left_mask)
                right_count = n_samples - left_count

                # min_samples_leaf 应在接受划分前检查，避免产生空子树。
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                weighted_impurity = (
                    left_count / n_samples * self._impurity_measure(y[left_mask])
                    + right_count
                    / n_samples
                    * self._impurity_measure(y[right_mask])
                )

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_trees(self, X, y, depth=0):
        """递归构建当前数据对应的子树。"""
        n_samples = y.size
        classes, counts = np.unique(y, return_counts=True)
        predict_class = classes[np.argmax(counts)]

        # 先建立叶子节点；只有满足继续划分条件时才补充划分信息和子树。
        node = Node(n_samples=n_samples, predict_class=predict_class)

        reached_max_depth = (
            self.max_depth is not None and depth >= self.max_depth
        )
        if (
            reached_max_depth
            or classes.size == 1
            or n_samples < self.min_samples_split
        ):
            return node

        feature_index, threshold = self._split_data(X, y)
        if feature_index is None:
            return node

        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_trees(
            X[left_mask],
            y[left_mask],
            depth=depth + 1,
        )
        node.right = self._build_trees(
            X[right_mask],
            y[right_mask],
            depth=depth + 1,
        )

        return node
    

    def score(self, X, y):
        """对结果进行打分，使用 accuracy scoring"""
        y_pred = self.predict(X)
        return accuracy_score(y,y_pred)

    def _predict_instance(self, x):
        """从根节点开始，为单个样本寻找对应的叶子节点。"""
        current = self._root

        while current.feature_index is not None:
            if x[current.feature_index] < current.threshold:
                current = current.left
            else:
                current = current.right

        return current.predict_class

    def _validate_hyperparameters(self):
        """检查控制树结构的超参数。"""
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, (int, np.integer))
            or self.max_depth < 0
        ):
            raise ValueError("max_depth must be None or a non-negative integer.")

        if (
            isinstance(self.min_samples_leaf, bool)
            or not isinstance(self.min_samples_leaf, (int, np.integer))
            or self.min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer.")

        if (
            isinstance(self.min_samples_split, bool)
            or not isinstance(self.min_samples_split, (int, np.integer))
            or self.min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer greater than 1.")

    def _as_float_array(self, values, name):
        """将特征转换为可用于阈值比较的浮点数组。"""
        try:
            return np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only numeric values.") from exc

    def _validate_training_data(self, X, y):
        """校验训练数据并返回格式统一的数组。"""
        X = self._as_float_array(X, "X")
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
            raise ValueError("y must not contain NaN or infinite values.")

        return X, y

    def _validate_prediction_data(self, X):
        """校验预测数据及其特征数量。"""
        X = self._as_float_array(X, "X")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must contain the same number of features as the training data."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")

        return X

    def _check_is_fitted(self):
        """检查模型是否已经构建完成。"""
        if self._root is None or self.n_features_in_ is None:
            raise ValueError("The model has not been fitted yet.")

    