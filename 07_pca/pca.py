import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components

        self.n_samples = None
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
    def fit(self,X):
        X = np.array(X)
        self.n_samples = len(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 计算协方差矩阵
        C = np.dot(X_centered.T, X_centered) / (self.n_samples - 1)
        # 计算特征值和特征向量
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(C)
        if len(self.eigenvalues) < self.n_components:
            raise ValueError("n_components is larger than eigen")
        indices = np.argsort(self.eigenvalues)[::-1][:self.n_components]
        self.eigenvalues = self.eigenvalues[indices]
        self.eigenvectors = self.eigenvectors[:, indices]

        return self

    def transform(self,X):
        X = np.array(X)
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)
