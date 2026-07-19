import numpy as np
from xgboost_tree import XGBoostTree

from utils import BaseEstimator


class XGBoost(BaseEstimator):
    def __init__(
        self, n_estimators=100, learning_rate=1, reg_lambda=1.0, gamma=0.0, max_depth=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_depth = max_depth

        self.estimators_ = []
        self.base_line = None

    def fit(self, X, y):

        self.estimators_.clear()

        # 初始化第 0 个模型，预测值都取平均值
        self.base_line = np.mean(y)
        gradients = y - self.base_line

        hessians = np.full(len(y), 1)
        for k in range(self.n_estimators):
            # 均方误差，二阶导 hessians 均为 1
            estimator = XGBoostTree(
                max_depth=self.max_depth, reg_lambda=self.reg_lambda, gamma=self.gamma
            )
            estimator.fit(X, gradients=gradients, hessians=hessians)
            self.estimators_.append(estimator)

            # 更新梯度
            y_pred = estimator.predict(X)
            gradients = gradients - y_pred

        return self

    def predict(self, X):
        y_pred = np.full(len(X), self.base_line)
        for k in range(self.n_estimators):
            y_pred = y_pred + self.learning_rate * self.estimators_[k].predict(X)

        return y_pred

        return y_pred
