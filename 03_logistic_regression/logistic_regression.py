import numpy as np

from utils import BaseEstimator, accuracy_score


class LogisticRegression(BaseEstimator):
    """
    逻辑回归，实现对二分类进行预测

    z = X @ weights + bias，再通过 sigmoid 函数把结果映射到
    0 到 1 之间，作为样本属于正类的概率。

    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        fit_intercept=True,
        threshold=0.5,
        tolerance=1e-6,
    ):
        """初始化逻辑回归模型。

        构造函数只负责保存超参数和初始化模型属性，不在这里执行训练。
        权重和偏置会在 `fit` 方法中根据训练数据的特征数量再正式创建。

        :param learning_rate: 梯度下降的学习率，控制每次参数更新的步长。
        :param epochs: 最大训练轮数，也就是最多进行多少次梯度更新。
        :param fit_intercept: 是否学习偏置项。如果为 False，模型只学习权重。
        :param threshold: 分类阈值。预测概率大于等于该值时输出正类 1。
        :param tolerance: 提前停止阈值，用于判断相邻两轮损失变化是否足够小。
        """
        # 训练过程相关的超参数。
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance

        # 模型结构和预测行为相关的超参数。
        self.fit_intercept = fit_intercept
        self.threshold = threshold

        # 训练后得到的模型参数。此处先置空，等 fit 时再根据特征数量初始化。
        self.weights = None
        self.bias = None

        # 记录每轮训练损失，方便后续观察模型是否收敛。
        self.loss_history = []

    def fit(self, X, y):
        """训练线性回归模型。

        :param X: 训练特征矩阵，形状通常为 (n_samples, n_features)。
        :param y: 训练目标值数组，形状通常为 (n_samples,)。
        :return: 训练后的模型自身。
        :raises ValueError: 当输入为空、维度不正确或样本数量不一致时抛出。
        """
        X, y = self._validate_prediction_data(X)

        sample_numbers = len(X)
        feature_numbers = len(y)

        self.weights = np.zeros(feature_numbers)
        self.bias = 0

        # 开始梯度学习
        last_loss = self._compute_loss(y, y_pred=self._compute_prediction(X))
        for _ in self.epochs:

            y_pred = self._compute_prediction(X)

            errors = y_pred - y

            dw = np.dot(X.T, errors)
            self.weight = self.weight - self.learning_rate * dw

            if self.fit_intercept:
                db = np.sum(errors) / sample_numbers
                self.bias = self.bias - self.learning_rate * db

            # 观察是否可以停止学习
            curr_loss = self._compute_loss(y, y_pred=self._compute_predictions(X))
            if np.abs(curr_loss - last_loss) <= self.tolerance:
                break
            last_loss = curr_loss

        return self

    def predict(self, X):
        """使用训练后的模型预测连续目标值。

        :param X: 待预测特征矩阵，形状通常为 (n_samples, n_features)。
        :return: 预测值数组，形状通常为 (n_samples,)。
        :raises ValueError: 当模型尚未训练或输入特征维度不匹配时抛出。
        """

        self._check_is_fitted()
        X = self._validate_prediction_data(X)
        y_pred = self._compute_predictions(X)
        return self._transform_probability_to_classification(y_pred)

    def _compute_prediction(self, X):
        """根据当前参数计算预测值。


        z = X @ weights + bias 采用 sigmoid 进行映射
        该方法通常被 `fit` 和 `predict` 复用，用来集中维护线性预测公式。

        :param X: 已整理好的二维特征矩阵。
        :return: 预测值数组，形状通常为 (n_samples,)。
        """

        Z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-Z))

    def _compute_loss(self, y_true, y_pred):
        """计算当前预测结果的损失函数
            使用交叉熵进行估计

        :param y_true: 真实目标值数组。
        :param y_pred: 模型预测值数组。
        :return: 当前损失值，通常为一个浮点数。
        """

        # 为了防止 y_pred 出现 0 或 1 导致 log(0) 报错，通常会做一个微小的截断（Clip）
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _transform_probability_to_classification(self, y_pred_probability):
        """
        将回归计算出来的概率转换为阴阳样本
        :param y_pred_probability: 模型预测的概率 尚未转换为阴阳样本，大小 (sample_numbers,)
        """

        return np.array([1 if p >= self.threshold else 0 for p in y_pred_probability])

    def score(self, X, y):
        """计算模型在给定数据上的回归得分。
        对于二元逻辑回归问题，我们采用 accuracy 对模型进行打分

        :param X: 验证或测试特征矩阵，形状通常为 (n_samples, n_features)。
        :param y: 真实目标值数组，形状通常为 (n_samples,)。
        :return: 模型得分，通常为一个浮点数。
        """
        self._validate_scoring_data(X, y)
        return accuracy_score(y, y_pred=self.predict(X))

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
            raise ValueError("X must contain the same number of features as the training data.")
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
