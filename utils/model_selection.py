import numpy as np

try:
    from .base_estimator import BaseEstimator
    from .metrics import accuracy_score, mean_squared_error, r2_score
except ImportError:
    from base_estimator import BaseEstimator
    from metrics import accuracy_score, mean_squared_error, r2_score


def train_test_split(X, y, test_size=0.2, shuffle=True, random_seed=None):
    """将数据集划分为训练集和测试集。

    :param X: 特征矩阵，形状为 (n_samples, n_features)。
    :param y: 标签或目标值数组，形状为 (n_samples,)。
    :param test_size: 测试集比例，取值范围为 [0, 1]，默认值为 0.2。
    :param shuffle: 是否在划分前打乱样本。
    :param random_seed: 随机种子，固定后可复现实验结果。
    :return: X_train, X_test, y_train, y_test
    """

    X = np.array(X)
    y = np.array(y)

    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    total_samples_number = X.shape[0]

    # 固定随机种子，保证同一份数据可以复现相同划分。
    if random_seed is not None:
        np.random.seed(random_seed)
    if shuffle:
        shuffled_indices = np.random.permutation(total_samples_number)
        X = X[shuffled_indices]
        y = y[shuffled_indices]

    test_number = int(total_samples_number * test_size)

    X_test = X[:test_number]
    y_test = y[:test_number]

    X_train = X[test_number:]
    y_train = y[test_number:]

    return X_train, X_test, y_train, y_test


# 交叉验证 KFold
class KFold:
    """
    交叉验证 KFlod
    核心思想是：
    不要只做一次训练集/验证集划分，而是把数据平均分成 K 份，每次拿其中 1 份做验证集，剩下 K-1 份做训练集，重复 K 次。
    """

    def __init__(self, n_splits=5, shuffle=True, random_seed=None):
        """
        初始化 KFlod

        :param n_splits: int 目标划分数
        :param shuffle: bool 是否打乱划分 默认为 True
        :param random_seed: 可选 随机种子

        """
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, X):
        """
        对矩阵 X 进行切分
        :param X: 特征矩阵，形状为(n_samples,n_features)
        """

        X = np.array(X)
        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot be greater than n_samples")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_seed)
            indices = rng.permutation(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1  # 处理不能被整除的样本

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size

            test_indices = indices[start:stop]  # 测试集
            train_indices = np.concatenate([indices[:start], indices[stop:]])  # 训练集

            yield train_indices, test_indices  # yield 语法，依次返回 (训练集，测试集)

            current = stop


# 网格搜索
class ParameterGrid:
    """
    网格搜索
    核心思想: 把想要尝试的超参数组合全部枚举一遍，然后用验证集或交叉验证选表现最好的那一组。
    """

    def __init__(self, param_dict):
        """
        初始化 ParameterGrid

        :param param_grid: dict 目标枚举超参字典
        """
        if not isinstance(param_dict, dict):
            raise TypeError("param_grid must be a dict")

        self.param_grid = param_dict

        for key, values in self.param_grid.items():
            if len(values) == 0:
                raise ValueError(f"param_grid[{key!r}] cannot be empty")

    def __iter__(self):
        """
        实现迭代器，可以直接 for 循环遍历这个类：
            for params in ParameterGrid(param_grid):
        """

        results = [{}]

        for key in self.param_grid:
            new_results = []

            for old_params in results:
                for value in self.param_grid[key]:
                    new_params = old_params.copy()
                    new_params[key] = value
                    new_results.append(new_params)

            results = new_results

        for params in results:
            yield params

    def __len__(self):
        """
        获取一共有多少组超参
        """
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total


class GridSearchCV:
    """
    网格搜索交叉验证工具。

    该类用于在给定的超参数搜索空间中枚举所有参数组合，
    并通过 K 折交叉验证评估每一组参数的表现，最终选择
    平均验证得分最高的一组参数作为最优参数。

    模型对象通常需要实现以下方法：
    - fit(X, y)：训练模型
    - predict(X)：进行预测
    - score(X, y)：可选，当 scoring=None 时可作为默认评分方式

    主要属性：
    - best_params_：最优参数组合
    - best_score_：最优平均验证得分
    - best_estimator_：使用最优参数重新训练后的模型
    - cv_results_：所有参数组合的交叉验证结果
    """

    def __init__(
        self,
        estimator,
        param_dict,
        cv=5,
        scoring="accuracy",
        shuffle=True,
        random_seed=None,
        refit=True,
    ):
        """
        初始化网格搜索交叉验证器。

        :param estimator: 待调参的模型对象
        :param param_dict: 参数搜索空间，格式为 {参数名: 参数候选值列表}
        :param cv: 交叉验证折数，默认 5
        :param scoring: 评分方式，支持 "accuracy"、"r2"、"neg_mse"、None 或自定义函数
        :param shuffle: 是否在划分前打乱数据，默认 True
        :param random_seed: 随机种子，用于复现实验结果
        :param refit: 是否使用最优参数在完整训练集上重新训练模型
        """
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("estimator must inherit from BaseEstimator.")

        if not isinstance(param_dict, dict):
            raise TypeError("param_grid must be a dict.")

        if not isinstance(cv, int) or cv < 2:
            raise ValueError("cv must be an integer >= 2.")

        self.cv = KFold(n_splits=cv, shuffle=shuffle, random_seed=random_seed)
        self.estimator = estimator
        self.param_grid = param_dict
        self.scoring = scoring
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.refit = refit

        self.cv_results_ = []
        self.cv_result_ = self.cv_results_
        self.best_params_ = None
        self.best_param_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def fit(self, X, y):
        """
        执行网格搜索和交叉验证。

        该方法会遍历所有参数组合，对每一组参数执行 K 折交叉验证，
        计算平均验证得分，并记录最优参数和完整搜索结果。

        :param X: 特征矩阵，形状为 (样本数, 特征数)
        :param y: 标签或目标值数组，形状为 (样本数,)
        :return: self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        best_score = -np.inf
        best_params = None
        self.cv_results_.clear()

        for params in ParameterGrid(self.param_grid):
            model = self._clone_estimator(self.estimator)
            model.set_params(**params)

            scores = self._cross_val_score(model, X, y)
            mean_score = np.mean(scores)

            self.cv_results_.append(
                {
                    "params": params,
                    "mean_score": mean_score,
                    "std_score": np.std(scores),
                    "scores": scores,
                }
            )

            if mean_score > best_score:
                best_params = params
                best_score = mean_score

        self.best_params_ = best_params
        self.best_param_ = best_params
        self.best_score_ = best_score

        if self.refit:
            self.best_estimator_ = self._clone_estimator(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        """
        使用最优模型进行预测。

        该方法通常需要先调用 fit，并且 refit=True，
        使 best_estimator_ 已经完成训练。

        :param X: 待预测的特征矩阵
        :return: 预测结果
        """
        if not self.refit:
            raise ValueError("refit must be True if you want to use the method")

        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before predict()")

        return self.best_estimator_.predict(X)

    def _clone_estimator(self, estimator=None):
        """
        克隆一个新的模型对象。

        每一折交叉验证都应使用新的模型对象，
        避免不同实验之间的训练状态互相影响。

        :return: 克隆后的模型对象
        """
        if estimator is None:
            estimator = self.estimator

        if not hasattr(estimator, "get_params"):
            raise TypeError("estimator must implement get_params()")

        params = estimator.get_params()
        return estimator.__class__(**params)

    def _get_scorer(self):
        """
        根据 scoring 获取评分函数。

        :return: 接收 (y_true, y_pred) 的评分函数
        """
        if self.scoring == "accuracy":
            return accuracy_score
        if self.scoring == "r2":
            return r2_score
        if self.scoring == "neg_mse":
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        if self.scoring is None:
            return None
        if callable(self.scoring):
            return self.scoring

        raise ValueError("scoring must be 'accuracy', 'r2', 'neg_mse', None, or a callable.")

    def _cross_val_score(self, estimator, X, y):
        """
        计算模型在验证集上的得分
        使用交叉验证取平均值

        :param estimator: 要训练的模型对象
        :param X_val: 验证集特征矩阵
        :param y_val: 验证集真实标签或目标值
        :return: list 验证集得分列表
        """
        X = np.array(X)
        y = np.array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        scores = []
        scorer = self._get_scorer()

        for train_indices, test_indices in self.cv.split(X):
            model = self._clone_estimator(estimator)

            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            model.fit(X_train, y_train)
            if scorer is None:
                scores.append(model.score(X_test, y_test))
            else:
                y_pred = model.predict(X_test)
                scores.append(scorer(y_true=y_test, y_pred=y_pred))

        return scores
