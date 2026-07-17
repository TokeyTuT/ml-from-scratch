from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_TREE_DIR = PROJECT_ROOT / "04_decision_tree"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DECISION_TREE_DIR))

from utils import BaseEstimator
from decision_tree_regression import DecisionTreeRegression


class GBDTRegression(BaseEstimator):
    def __init__(self,n_estimators = 100,learning_rate = 0.1,max_depth = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.estimators_ = [] # 存储每次的弱学习树
        self.base_line = None 
    def fit(self, X, y):
        estimator.clear()
        # 初始化模型，预测值为真实值的平均值
        self.base_line = np.mean(y)
        y = np.array([yt - self.base_line for yt in y]) # 伪梯度：真实值 - 预测值

        for k in range(self.n_estimators):
            estimator,y = self._weak_decision_tree(X,y)
            self.estimators_.append(estimator)  
        return self
    
    def predict(self, X):
        y_pred = np.full(len(X),self.base_line)

        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)

        return y_pred
    
    def _weak_decision_tree(self,X,y):
        estimator = DecisionTreeRegression(max_depth=self.max_depth)
        estimator.fit(X,y)

        y_pred = estimator.predict(X)

        errors = y - y_pred
        return estimator,errors


