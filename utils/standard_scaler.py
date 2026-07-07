import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.variane_ = None # 方差
        self.scale_ = None # 标准差

    def fit(self,X):
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        
        self.mean_ = np.mean(X,axis = 0)
        self.variane_ = np.var(X,axis = 0)
        self.scale_ = np.std(X,axis = 0)
        self.scale_ = np.where(self.scale_ == 0,1,self.scale_) # 防止被 0 整除出现异常值

    def transform(self,X):
        self._check_is_fitted()

        X = np.array(X)

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("The number of features in X_scaled must be the same as that during the fit process.")
        
        return (X - self.mean_) / self.scale_

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self,X_scaled):
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
        if self.mean_ is None:
            raise ValueError("StandardScaler has not been fitted yet. Please call fit or fit_transform first.")
        


# if __name__ == "__main__":
#     X = np.array([
#         [1, 10],
#         [2, 20],
#         [3, 30],
#         [4, 40],
#     ], dtype=float)

#     scaler = StandardScaler()
#     scaler.fit(X)

#     X_scaled = scaler.transform(X)

#     print("原始数据:")
#     print(X)

#     print("\n均值:")
#     print(scaler.mean_)

#     print("\n标准差:")
#     print(scaler.scale_)

#     print("\n标准化后的数据:")
#     print(X_scaled)

#     print("\n标准化后每一列的均值，应接近 0:")
#     print(np.mean(X_scaled, axis=0))

#     print("\n标准化后每一列的标准差，应接近 1:")
#     print(np.std(X_scaled, axis=0))

#     print("\n均值是否接近 0:")
#     print(np.allclose(np.mean(X_scaled, axis=0), np.array([0.0, 0.0])))

#     print("\n标准差是否接近 1:")
#     print(np.allclose(np.std(X_scaled, axis=0), np.array([1.0, 1.0])))

#     X_original = scaler.inverse_transform(X_scaled)

#     print("\n还原后的数据:")
#     print(X_original)

#     print("\n还原结果是否等于原始数据:")
#     print(np.allclose(X_original, X))
