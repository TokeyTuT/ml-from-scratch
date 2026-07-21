import numpy as np

from utils import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, K=2, n_ranges=3, random_seed=None):
        self.K = K
        self.random_seed = random_seed
        self.n_ranges = n_ranges
        self.mus = None
        self.clusters = None

    def fit(self, X):
        # 随机选 K 个样本初始化中心矩阵
        rng = np.random.default_rng(self.random_seed)
        indices = rng.choice(X.shape[0], size=self.K, replace=False)
        mus = X[indices].astype(float)

        for _ in range(self.n_ranges):
            # 将每个样本分配给距离最近的中心
            distances = np.array([
                [self._calc_distance(x, mu) for mu in mus]
                for x in X
            ])
            clusters = np.argmin(distances, axis=1)

            # 更新中心参数
            new_mus = mus.copy()
            for i in range(self.K):
                cluster_samples = X[clusters == i]
                if len(cluster_samples) > 0:
                    new_mus[i] = np.mean(cluster_samples, axis=0)

            if np.allclose(mus, new_mus):
                mus = new_mus
                break

            mus = new_mus

        self.mus = mus
        self.clusters = self.predict(X)
        return self

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        cluster = None
        dis = np.inf
        for i, mu in enumerate(self.mus):
            current_dis = self._calc_distance(x, mu)
            if current_dis < dis:
                dis = current_dis
                cluster = i
        return cluster

    def _calc_distance(self, x, mu):
        return np.sum((x - mu) ** 2)
