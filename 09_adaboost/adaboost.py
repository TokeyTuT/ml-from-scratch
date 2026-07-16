import numpy as np
from utils import BaseEstimator

class WeakClassifier:
    def __init__(self,feature,threshold,error,polarity):
        self.feature = feature
        self.threshold = threshold
        self.error = error
        self.polarity = polarity

    def predict(self,x):
        """
        x 是单个样本
        """
        if x[self.feature] < self.threshold:
            return self.polarity

        return -self.polarity

class AdaBoost(BaseEstimator):
    def __init__(self,n_estimators = 50):

        self.n_estimators = n_estimators

        self.n_samples = None # 样本数量
        self.model_weights = [] # 模型的权重
        self.feature_weights = None # 样本的权重
        self.estimators = []

    def fit(self, X, y):
        """
        :param y:默认传入的标签是一个二分类,形状为(n_samples,),数据类型为 int 且只有 -1 和 +1
        """
        self.n_samples = len(y)
        self.feature_weights = np.full(self.n_samples, 1.0 / self.n_samples)
        self.model_weights = []
        self.estimators = []

        for _ in range(self.n_estimators):
            self.estimators.append(self._weak_classifier_processing(X,y))

        return self

    def predict(self, X):        
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self,x):
        # 权重投票选举
        Hx = 0
        for i,estimator in enumerate(self.estimators):
            flag = estimator.predict(x)
            Hx += flag * self.model_weights[i]

        return 1 if Hx > 0 else -1


    def _find_best_feature(self,X,y):
        feature_numbers = X.shape[1]

        best_feature = None
        best_error = np.inf
        best_threshold = None
        best_polarity = None

        # 寻找最佳特征类
        for j in range(feature_numbers):
            # 寻找当前特征值的最佳分割点
            X_j = X[:,j]
            X_sort = np.sort(X_j)

            for p in range(0,self.n_samples - 1):
                value = (X_sort[p] + X_sort[p+1]) / 2

                # 分别检查左右两种分类方向
                for polarity in (1,-1):
                    predictions = np.where(
                        X_j < value,
                        polarity,
                        -polarity
                    )
                    error = np.sum(
                        self.feature_weights[predictions != y]
                    )

                    if error < best_error:
                        best_feature = j
                        best_error = error
                        best_threshold = value
                        best_polarity = polarity

        return best_feature,best_error,best_threshold,best_polarity
    
    def _weak_classifier_processing(self,X,y):

        # 找到最佳点
        (best_feature,
        best_error,
        best_threshold,
        best_polarity) = self._find_best_feature(X,y)

        # 更新模型权重
        a = 0.5 * np.log((1-best_error)/best_error)
        self.model_weights.append(a)

        weak_estimator = WeakClassifier(
            best_feature,
            best_threshold,
            best_error,
            best_polarity
        )
        predictions = np.array([
            weak_estimator.predict(x) for x in X
        ])

        # 更新并归一化样本权重
        self.feature_weights *= np.exp(-a * y * predictions)
        self.feature_weights /= np.sum(self.feature_weights)
        
        return weak_estimator


        
        

        
            
