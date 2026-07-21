import numpy as np
from utils import BaseEstimator


class NaiveBayes(BaseEstimator):
    def __init__(self, laplace_smoothing_index=0):
        self.laplace_smoothing_index = laplace_smoothing_index

        self.n_features = None
        self._prior_possible = {}  # 字典
        self._likelihood = {}  # { C_j:[] }
        self._feature_values = []

    def fit(self, X, y):
        self._prior_possible.clear()
        self._likelihood.clear()

        n_samples, self.n_features = X.shape
        # 计算先验概率
        unique_y, counts = np.unique(y, return_counts=True)
        self._prior_possible = dict(zip(unique_y, counts / len(y)))
        self._feature_values = [
            np.unique(X[:, feature]) for feature in range(self.n_features)
        ]
        # 计算似然概率
        for c in unique_y:
            self._likelihood[c] = list()
            mask = y == c
            for feature in range(self.n_features):
                X_feature = X[mask, feature]
                unique_X, counts_X = np.unique(X_feature, return_counts=True)
                feature_counts = dict(zip(unique_X, counts_X))
                all_feature_values = self._feature_values[feature]
                denominator = len(X_feature) + (
                    self.laplace_smoothing_index * len(all_feature_values)
                )
                feature_dict = {
                    value: (
                        feature_counts.get(value, 0) + self.laplace_smoothing_index
                    )
                    / denominator
                    for value in all_feature_values
                }
                self._likelihood[c].append(feature_dict)

        return self

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        prediction = None
        possible = -1

        for c, prior_possible in self._prior_possible.items():
            current_possible = prior_possible
            for feature in range(self.n_features):
                current_possible *= self._likelihood[c][feature].get(x[feature], 0)

            if current_possible > possible:
                possible = current_possible
                prediction = c

        return prediction
