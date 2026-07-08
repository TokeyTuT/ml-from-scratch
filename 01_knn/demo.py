import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import KNNClassifier
from model import KNNRegression



def test_classifier():
    """验证 KNN 分类器在二维玩具数据上的预测结果。"""
    x_train = np.array([
            [0, 0],
            [1, 1],
            [10, 10],
            [11, 11],])
    y_train = np.array([0,0,1,1])

    x_test = np.array([
        [0.5, 0.5],
        [10.5, 10.5],
        [3, 3],
    ])

    estimator = KNNClassifier()
    estimator.fit(x_train,y_train)

    res = estimator.predict(x_test)

    print("分类问题：", res)


def test_regression():
    """验证 KNN 回归器在二维玩具数据上的预测结果。"""
    x_train = np.array([
    [0, 0],
    [1, 1],
    [2, 2],
    [10, 10],
    [11, 11],
    [12, 12],
    ])

    y_train = np.array([
    0.0,
    1.0,
    2.0,
    10.0,
    11.0,
    12.0,
    ])

    x_test = np.array([
    [0.5, 0.5],
    [10.5, 10.5],
    [3.0, 3.0],
    ])  


    estimator = KNNRegression(3)
    estimator.fit(x_train,y_train)
    res = estimator.predict(x_test)

    print("回归问题:" ,res)

def main():
    test_classifier()
    test_regression()


if __name__ == "__main__":
    main()
