import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from utils import StandardScaler,train_test_split,root_mean_squared_error,r2_score
from model import KNNRegression

DATA_PATH = PROJECT_ROOT / "datasets" / "california_housing_sample.csv"


def demo():
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

    # 对数据进行标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 调用模型
    estimator = KNNRegression()
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)

    rmse = root_mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print("RMSE ：",rmse)
    print("R^2：",r2)


if __name__ == "__main__":
    demo()
