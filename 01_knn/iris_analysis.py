from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from knn import KNNClassifier
from utils import MinMaxScaler, train_test_split, accuracy_score


DATA_PATH = PROJECT_ROOT / "datasets" / "iris.csv"

def demo():
    """使用鸢尾花数据集验证 KNN 分类器。"""
    data = np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        dtype=str,
        skip_header=1
    )

    # 只使用前三个数值特征，标签位于最后一列。
    X = data[:,:3].astype(dtype=float)
    y = data[:,4]

    # 划分训练集和测试集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

    # KNN 对特征尺度敏感，预测前先做归一化。
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    estimator = KNNClassifier(n_neighbors=3)
    estimator.fit(X_train,y_train)

    y_pred = estimator.predict(X_test)

    score = accuracy_score(y_test,y_pred)
    print("模型的准确率为:",score)

if __name__ == "__main__":
    demo()
