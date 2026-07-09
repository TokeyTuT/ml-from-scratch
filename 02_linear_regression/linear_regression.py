import numpy as np

from utils import BaseEstimator
from utils import mean_squared_error, r2_score


class LinearRegression(BaseEstimator):
    """线性回归模型。

    该类用于从零实现连续值预测任务中的线性回归模型。
    模型假设特征和目标值之间满足近似线性关系：

        y_hat = X @ weights + bias

    当前文件只定义推荐接口和每个方法的职责，不包含训练、
    预测、损失计算或梯度计算的具体实现。

    :ivar weights: 模型权重，训练后形状应为 (n_features,)。
    :ivar bias: 模型偏置项，训练后通常为浮点数。
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        fit_intercept=True,
        tolerance=1e-12,
        solver="granient_descent" # 此处暂时用梯度下降法，正规方程法先鸽一会
    ):
        """初始化线性回归模型。

        这里只保存超参数和训练后会用到的属性，不执行训练逻辑。
        具体参数初始化方式可以在 `_initialize_parameters` 中完成。

        :param learning_rate: 梯度下降的学习率，控制每次参数更新步长。
        :param epochs: 最大训练轮数。
        :param fit_intercept: 是否学习偏置项。如果为 False，模型只学习权重。
        :param tolerance: 提前停止阈值。可用于判断相邻两轮损失变化是否足够小。
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.solver = solver

        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """训练线性回归模型。

        :param X: 训练特征矩阵，形状通常为 (n_samples, n_features)。
        :param y: 训练目标值数组，形状通常为 (n_samples,)。
        :return: 训练后的模型自身。
        :raises ValueError: 当输入为空、维度不正确或样本数量不一致时抛出。
        """
        X, y = self._validate_training_data(X, y)

        sample_numbers = len(y)
        feature_numbers = X.shape[1]
        
        # 初始化参数偏置和权重全都初始化为 0
        self.bias = 0
        self.weights = np.zeros(feature_numbers)
        
        last_loss = self._compute_loss(y,y_pred=self._compute_predictions(X))
        for _ in range(self.epochs):
            y_pred = self._compute_predictions(X)
            errors = y_pred - y

            # 考虑矩阵的乘法: dw = X^T errors 注意我们最终答案是要得到 形状为 (feature,) 数组，不要把这个乘法写反了
            # 我们可以直接得权重的偏导值向量
            dw = np.dot(X.T,errors) / sample_numbers
            self.weights = self.weights - self.learning_rate * dw
            
            if self.fit_intercept:
                db = np.sum(errors) / sample_numbers
                self.bias = self.bias - self.learning_rate * db

            # 观察是否可以停止学习
            curr_loss = self._compute_loss(y,y_pred=self._compute_predictions(X))
            if np.abs(curr_loss - last_loss) <= self.tolerance:
                break
            last_loss = curr_loss
                
            # 未优化版本
            # for j in range(feature_numbers):
            #     dw = 0
            #     for i in range(sample_numbers):
            #         dw = dw + (self._compute_predictions(X[i]) - self.bias)*X[i][j]
            #     dw = dw/sample_numbers
            #     self.weights[j] = self.weights[j] - self.learning_rate * dw

            # # 更新偏置
            # if self.fit_intercept:
            #     db = (self._compute_predictions(X[i]) - self.bias)/m
        
        return self

    def predict(self, X):
        """使用训练后的模型预测连续目标值。

        :param X: 待预测特征矩阵，形状通常为 (n_samples, n_features)。
        :return: 预测值数组，形状通常为 (n_samples,)。
        :raises ValueError: 当模型尚未训练或输入特征维度不匹配时抛出。
        """
        self._check_is_fitted()
        X = self._validate_prediction_data(X)
        return self._compute_predictions(X)
    

    def score(self, X, y):
        """计算模型在给定数据上的回归得分 R^2。

        :param X: 验证或测试特征矩阵，形状通常为 (n_samples, n_features)。
        :param y: 真实目标值数组，形状通常为 (n_samples,)。
        :return: 模型得分，通常为一个浮点数。
        """
        self._check_is_fitted()
        X, y = self._validate_scoring_data(X, y)
        y_pred = self._compute_predictions(X)
        return r2_score(y, y_pred=y_pred)


    def _compute_predictions(self, X):
        """根据当前参数计算预测值。

        该方法通常被 `fit` 和 `predict` 复用，用来集中维护线性预测公式。

        :param X: 已整理好的二维特征矩阵。
        :return: 预测值数组，形状通常为 (n_samples,)。
        """
        return np.array(np.dot(X, self.weights) + self.bias)

    def _compute_loss(self, y_true, y_pred):
        """计算当前预测结果的损失 MSE。

        :param y_true: 真实目标值数组。
        :param y_pred: 模型预测值数组。
        :return: 当前损失值，通常为一个浮点数。
        """
        return mean_squared_error(y_true, y_pred)


    def _as_float_array(self, values, name):
        """将输入转换为浮点数组，便于后续矩阵计算。"""
        try:
            return np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only numeric values.") from exc

    def _validate_training_data(self, X, y):
        """校验训练数据，并返回可直接用于训练的浮点数组。"""
        X = self._as_float_array(X, "X")
        y = self._as_float_array(y, "y")

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
        if not np.all(np.isfinite(y)):
            raise ValueError("y must not contain NaN or infinite values.")

        return X, y

    def _validate_prediction_data(self, X):
        """校验预测数据，并确保特征数量和训练时一致。"""
        X = self._as_float_array(X, "X")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "X must contain the same number of features as the training data."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")

        return X

    def _validate_scoring_data(self, X, y):
        """校验评分数据，避免指标函数收到形状不匹配的输入。"""
        X = self._validate_prediction_data(X)
        y = self._as_float_array(y, "y")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must not contain NaN or infinite values.")

        return X, y

    def _check_is_fitted(self):
        """检查模型是否已经训练完成。

        :return: None。
        :raises ValueError: 当 `weights` 或必要的模型参数尚未初始化时抛出。
        """
        if self.weights is None or self.bias is None:
            raise ValueError("The model has not be fitted yet.")
