import numpy as np


class MinMaxScaler:
    """Min-Max 特征归一化器。"""

    def __init__(self, feature_range=(0, 1)):
        """初始化归一化器。

        :param feature_range: 目标缩放区间，默认值为 (0, 1)。
        """

        if feature_range[0] >= feature_range[1]:
            raise ValueError("The first value of feature_range must be less than the second value.")
        self.feature_range = feature_range

        self.max_ = None
        self.min_ = None
        self.scaler_parameter = None
        
        self.data_range = None
        
    def fit(self, X):
        """计算每个特征的最小值和最大值。

        该方法只从数据中学习缩放参数，不直接返回缩放后的数据。

        :param X: 用于计算缩放参数的训练数据，形状为 (n_samples, n_features)。
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")

        self.max_ = np.max(X,axis = 0)
        self.min_ = np.min(X,axis = 0)

        self.data_range = self.max_- self.min_
        # 常数列的范围为 0，替换为 1 以避免除零。
        self.data_range = np.where(self.data_range == 0, 1, self.data_range)
        self.scaler_parameter = (self.feature_range[1] - self.feature_range[0]) / (self.data_range)
        

    def transform(self, X):
        """使用 fit 学到的参数缩放输入数据。

        :param X: 待缩放数据，形状为 (n_samples, n_features)。
        :return: 缩放后的数据，形状为 (n_samples, n_features)。
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")

        if X.shape[1] != self.max_.shape[0]:
            raise ValueError("The number of features in X_scaled must be the same as that during the fit process.")

        return (X - self.min_)*self.scaler_parameter + self.feature_range[0]
        

    def fit_transform(self, X):
        """先学习缩放参数，再返回缩放后的数据。

        :param X: 用于拟合并缩放的数据，形状为 (n_samples, n_features)。
        :return: 缩放后的数据，形状为 (n_samples, n_features)。
        """

        X = np.array(X)
        self.fit(X)
        return self.transform(X)


    def inverse_transform(self, X_scaled):
        """将缩放后的数据还原到原始特征尺度。
 
        :param X_scaled: 已经缩放的数据，形状为 (n_samples, n_features)。
        :return: 还原到原始尺度后的数据，形状为 (n_samples, n_features)。
        """
        self._check_is_fitted()

        X_scaled = np.asarray(X_scaled, dtype=float)

        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        if X_scaled.ndim != 2:
            raise ValueError("X_scaled must be a 2D array.")

        if X_scaled.shape[1] != self.min_.shape[0]:
            raise ValueError("The number of features in X_scaled must be the same as that during the fit process.")

        return (X_scaled - self.feature_range[0]) / self.scaler_parameter + self.min_

    def _check_is_fitted(self):
        """检查当前缩放器是否已经调用过 fit。"""
        if self.min_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet. Please call fit or fit_transform first.")
        


if __name__ == "__main__":

    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
        [40, 400],
    ], dtype=float)

    expected_scaled = np.array([
        [0.0,        0.0],
        [1 / 3,      1 / 3],
        [2 / 3,      2 / 3],
        [1.0,        1.0],
    ])
    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    print(X_scaled)
    print(np.allclose(X_scaled, expected_scaled))

    X_original = scaler.inverse_transform(X_scaled)

    print(X_original)
    print(np.allclose(X_original, X))
