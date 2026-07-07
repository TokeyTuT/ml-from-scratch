import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from model import KNNClassifier
from utils import MinMaxScaler, train_test_split, accuracy_score


DATA_PATH = PROJECT_ROOT / "datasets" / "iris.csv"

def demo():
    data = np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        dtype=str,
        skip_header=1
    )

    # 数据预处理
    X = data[:,:3].astype(dtype=float)
    y = data[:,4]

    # 划分训练集和测试集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

    # 数据集预处理
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN
    estimator = KNNClassifier(n_neighbors=3)
    estimator.fit(X_train,y_train)

    y_pred = estimator.predict(X_test)

    score = accuracy_score(y_test,y_pred)
    print("模型的准确率为:",score)

if __name__ == "__main__":
    # print("hello world!")
    demo()
