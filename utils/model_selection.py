import numpy as np


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


# 交叉验证 KFlod
class KFlod:
    """
    交叉验证 KFlod 
    核心思想是：
    不要只做一次训练集/验证集划分，而是把数据平均分成 K 份，每次拿其中 1 份做验证集，剩下 K-1 份做训练集，重复 K 次。
    """
    def __init__(
            self,
            n_splits=5,
            shuffle=True,
            random_seed=None
            ):
        if not isinstance(n_splits,int) or n_split >= 2:
            raise ValueError("n_splits must be an integer >= 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self,X):
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
        
        fold_size = np.full(self.n_splits,n_samples//self.n_splits)


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
        para_dict,
        cv=5,
        scoring=None,
        shuffle=True,
        random_seed=None,
        refit=True,
        verbose=False,
    ):
        """
        初始化网格搜索交叉验证器。

        :param estimator: 待调参的模型对象
        :param para_dict: 参数搜索空间，格式为 {参数名: 参数候选值列表}
        :param cv: 交叉验证折数，默认 5
        :param scoring: 评分函数，默认 None
        :param shuffle: 是否在划分前打乱数据，默认 True
        :param random_seed: 随机种子，用于复现实验结果
        :param refit: 是否使用最优参数在完整训练集上重新训练模型
        :param verbose: 是否输出搜索过程信息
        """
        self.estimator = estimator
        



    def fit(self, X, y):
        """
        执行网格搜索和交叉验证。

        该方法会遍历所有参数组合，对每一组参数执行 K 折交叉验证，
        计算平均验证得分，并记录最优参数和完整搜索结果。

        :param X: 特征矩阵，形状为 (样本数, 特征数)
        :param y: 标签或目标值数组，形状为 (样本数,)
        :return: self
        """
        pass

    def predict(self, X):
        """
        使用最优模型进行预测。

        该方法通常需要先调用 fit，并且 refit=True，
        使 best_estimator_ 已经完成训练。

        :param X: 待预测的特征矩阵
        :return: 预测结果
        """
        pass

    def _generate_param_combinations(self, para_dict):
        """
        根据参数网格生成所有参数组合。

        :param para_dict: 参数搜索空间字典
        :return: 参数组合列表，每个元素都是一个参数字典
        """
        pass

    def _split_k_fold(self, X, y):
        """
        生成 K 折交叉验证的训练集和验证集索引。

        :param X: 特征矩阵
        :param y: 标签或目标值数组
        :return: K 折索引列表，每个元素为 (train_indices, val_indices)
        """
        pass

    def _clone_estimator(self):
        """
        克隆一个新的模型对象。

        每一折交叉验证都应使用新的模型对象，
        避免不同实验之间的训练状态互相影响。

        :return: 克隆后的模型对象
        """
        pass

    def _set_params(self, model, params):
        """
        为模型设置当前参数组合。

        :param model: 待设置参数的模型对象
        :param params: 当前参数组合字典
        :return: None
        """
        pass

    def _score(self, model, X_val, y_val):
        """
        计算模型在验证集上的得分。

        如果传入 scoring，则使用 scoring 计算分数；
        否则可以尝试调用模型自身的 score 方法。

        :param model: 已训练好的模型对象
        :param X_val: 验证集特征矩阵
        :param y_val: 验证集真实标签或目标值
        :return: 验证集得分
        """
        pass
