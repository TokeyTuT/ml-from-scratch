from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from utils import StandardScaler,train_test_split,root_mean_squared_error,r2_score
from knn import KNNRegression

DATA_PATH = PROJECT_ROOT / "datasets" / "california_housing_sample.csv"


def demo():
    """使用加州房价样例数据验证 KNN 回归器。"""
    data = np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        dtype=float,
        skip_header=1
    )

    X = data[:,:-1]
    y = data[:,-1]
    
    # 划分数据集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # KNN 回归依赖距离计算，训练前先做标准化。
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    estimator = KNNRegression()
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)

    rmse = root_mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print("RMSE ：",rmse)
    print("R^2：",r2)


if __name__ == "__main__":
    demo()
