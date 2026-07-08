import numpy as np


class StandardScaler:
    """Z-Score 特征标准化器。"""

    def __init__(self):
        self.mean_ = None
        self.variane_ = None  # 方差
        self.scale_ = None  # 标准差

    def fit(self, X):
        """计算每个特征的均值和标准差。

        :param X: 训练数据，形状为 (n_samples, n_features)。
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        
        self.mean_ = np.mean(X,axis = 0)
        self.variane_ = np.var(X,axis = 0)
        self.scale_ = np.std(X,axis = 0)
        # 常数列的标准差为 0，替换为 1 以避免除零。
        self.scale_ = np.where(self.scale_ == 0,1,self.scale_)

    def transform(self, X):
        """使用 fit 学到的均值和标准差进行标准化。

        :param X: 待标准化数据，形状为 (n_samples, n_features)。
        :return: 标准化后的数据，形状为 (n_samples, n_features)。
        """
        self._check_is_fitted()

        X = np.array(X)

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("The number of features in X_scaled must be the same as that during the fit process.")
        
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """先学习标准化参数，再返回标准化后的数据。"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """将标准化后的数据还原到原始特征尺度。"""
        self._check_is_fitted() 
        X_scaled = np.asarray(X_scaled, dtype=float)

        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        if X_scaled.ndim != 2:
            raise ValueError("X_scaled must be a 2D array.")

        if X_scaled.shape[1] != self.mean_.shape[0]:
            raise ValueError("The number of features in X_scaled must be the same as that during the fit process.")
        

        return X_scaled * self.scale_ + self.mean_

    def _check_is_fitted(self):
        """检查当前标准化器是否已经调用过 fit。"""
        if self.mean_ is None:
            raise ValueError("StandardScaler has not been fitted yet. Please call fit or fit_transform first.")
