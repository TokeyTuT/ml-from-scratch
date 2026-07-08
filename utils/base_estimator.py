from abc import ABC, abstractmethod
import inspect


class BaseEstimator(ABC):
    """
    模型基础接口类。

    该类用于统一所有模型的基本接口，方便 GridSearchCV、
    交叉验证、评估工具等通用模块调用不同模型。

    子类必须实现：
    - fit(X, y)
    - predict(X)

    子类通常需要在 __init__ 中把超参数保存为同名属性，
    例如：
        self.n_neighbors = n_neighbors
        self.metric = metric

    这样 get_params 和 set_params 才能自动读取和修改参数。
    """

    @abstractmethod
    def fit(self, X, y):
        """
        训练模型。

        :param X: 特征矩阵
        :param y: 标签或目标值数组
        :return: self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        使用模型进行预测。

        :param X: 待预测特征矩阵
        :return: 预测结果
        """
        pass

    def get_params(self):
        """
        获取模型初始化参数。

        默认根据子类 __init__ 方法的参数名，读取同名实例属性。
        因此子类需要保证 __init__ 参数和实例属性同名。

        :return: 参数字典
        """
        signature = inspect.signature(self.__init__)
        params = {}

        for name in signature.parameters:
            if name == "self":
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)

        return params

    def set_params(self, **params):
        """
        设置模型参数。

        :param params: 参数名和值
        :return: self
        """
        valid_params = self.get_params()

        for name, value in params.items():
            if name not in valid_params:
                raise ValueError(f"Invalid parameter: {name}")
            setattr(self, name, value)

        return self

    def score(self, X, y):
        """
        计算模型得分。

        默认不实现，因为分类和回归的默认指标不同。
        分类模型可以重写为 accuracy，回归模型可以重写为 R2。

        :param X: 特征矩阵
        :param y: 真实标签或目标值
        :return: 模型得分
        """
        raise NotImplementedError("score method is not implemented.")
