import numpy as np

class MinMaxScaler:
    def __init__(self,feature_range = (0,1)):
        '''
        初始化 MinMaxScaler 默认下标映射范围 为0 ~ 1
        :param feature_range: tuple, 下标映射的范围，默认值为(0,1)
        '''

        if feature_range[0] >= feature_range[1]:
            raise ValueError("The first value of feature_range must be less than the second value.")
        self.feature_range = feature_range

        self.max_ = None
        self.min_ = None
        self.scaler_parameter = None
        data_range = self.max_ - self.min_
        self.data_range = np.where(data_range == 0, 1, data_range) # 防止 min == max 导致相除崩溃

    def fit(self,X):
        '''
        计算每个特征的最小值和最大值。

        这个方法只从数据中学习缩放参数，不会对数据进行缩放
        :param X : array-like, shape = (n_samples, n_features) 用于计算每个特征最小值和最大值的训练数据。
        '''
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")

        self.max_ = np.max(X,axis = 0)
        self.min_ = np.min(X,axis = 0)

        self.scaler_parameter = (self.feature_range[1] - self.feature_range[0]) / ()

        

    def transform(self,X):
        '''
        使用 fit 学习到的缩放参数，对输入数据进行缩放
        :param X : array-like, shape = (n_samples, n_features)
        需要被缩放的数据。
        :return X_scaled:  ndarray, shape = (n_samples, n_features) 缩放后的数据。
        '''
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")

        if X.shape[1] != self.max_.shape[0]:
            raise ValueError("The number of features in X_scaled must be the same as that during the fit process.")

        return (X - self.min_)*self.scaler_parameter + self.feature_range[0]
        

    def fit_transform(self,X):
        '''
        先根据输入数据计算缩放参数，然后返回缩放后的数据。
        :param X : array-like, shape = (n_samples, n_features)
            用于拟合并进行缩放的数据。

        :return X_scaled : ndarray, shape = (n_samples, n_features)
            缩放后的数据。
        '''

        X = np.array(X)
        self.fit(X)
        return self.transform(X)


    def inverse_transform(self, X_scaled):
        """
        将缩放后的数据还原到原始特征尺度。
 
        :param X_scaled : array-like, shape = (n_samples, n_features)
            已经经过 MinMaxScaler 缩放的数据。
        :return X_original : ndarray, shape = (n_samples, n_features)
            还原到原始尺度后的数据。
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
        """
        检查当前缩放器是否已经调用过 fit。
        """
        if self.min_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet. Please call fit or fit_transform first.")
        


# if __name__ == "__main__":

#     X = np.array([
#         [10, 100],
#         [20, 200],
#         [30, 300],
#         [40, 400],
#     ], dtype=float)

#     expected_scaled = np.array([
#         [0.0,        0.0],
#         [1 / 3,      1 / 3],
#         [2 / 3,      2 / 3],
#         [1.0,        1.0],
#     ])
#     scaler = MinMaxScaler()

#     X_scaled = scaler.fit_transform(X)

#     print(X_scaled)
#     print(np.allclose(X_scaled, expected_scaled))

#     X_original = scaler.inverse_transform(X_scaled)

#     print(X_original)
#     print(np.allclose(X_original, X))

