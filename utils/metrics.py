import numpy as np


def accuracy_score(y_true, y_pred):
    """计算分类准确率。"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    
    return (y_true == y_pred).mean()



def mean_absolute_error(y_true, y_pred):
    """计算平均绝对误差 MAE。"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    
    return np.mean(np.abs(y_true - y_pred))
    

def mean_squared_error(y_true, y_pred):
    """计算均方误差 MSE。"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    
    return np.mean((y_pred - y_true) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """计算均方根误差 RMSE。"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    return np.sqrt(mean_squared_error(y_true,y_pred))


def r2_score(y_true, y_pred):
    """
    决定系数 R^2
    R^2 = 1：完美拟合。
    R^2 = 0：模型的预测效果和直接拿“平均值”当预测值一模一样。
    0 < R^2 < 1：数值越接近 1，说明模型的拟合效果越好，解释能力越强。
    R^2 < 0 : 说明模型选错了或者完全不适用。
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    # 残差平方和，表示模型预测误差。
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 总平方和，表示真实值相对均值的波动。
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # 当真实值全部相等时，分母为 0，需要单独处理。
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
        
    return 1.0 - (ss_res / ss_tot)  


def validate_metrics():
    """
    使用固定样例验证 metrics.py 中各个指标函数是否符合预期。

    :return: bool, 所有检查通过返回 True，否则返回 False。
    """
    passed = True

    value_checks = [
        {
            "name": "accuracy_score",
            "func": accuracy_score,
            "y_true": [1, 0, 1, 1, 0],
            "y_pred": [1, 0, 0, 1, 1],
            "expected": 0.6,
        },
        {
            "name": "mean_absolute_error",
            "func": mean_absolute_error,
            "y_true": [3.0, -0.5, 2.0, 7.0],
            "y_pred": [2.5, 0.0, 2.0, 8.0],
            "expected": 0.5,
        },
        {
            "name": "mean_squared_error",
            "func": mean_squared_error,
            "y_true": [3.0, -0.5, 2.0, 7.0],
            "y_pred": [2.5, 0.0, 2.0, 8.0],
            "expected": 0.375,
        },
        {
            "name": "root_mean_squared_error",
            "func": root_mean_squared_error,
            "y_true": [3.0, -0.5, 2.0, 7.0],
            "y_pred": [2.5, 0.0, 2.0, 8.0],
            "expected": np.sqrt(0.375),
        },
        {
            "name": "r2_score",
            "func": r2_score,
            "y_true": [3.0, -0.5, 2.0, 7.0],
            "y_pred": [2.5, 0.0, 2.0, 8.0],
            "expected": 0.9486081370449679,
        },
    ]

    for check in value_checks:
        try:
            actual = check["func"](check["y_true"], check["y_pred"])
            if np.allclose(actual, check["expected"]):
                print(f"[PASS] {check['name']}")
            else:
                passed = False
                print(
                    f"[FAIL] {check['name']}: "
                    f"expected {check['expected']}, got {actual}"
                )
        except Exception as exc:
            passed = False
            print(f"[FAIL] {check['name']}: raised {type(exc).__name__}: {exc}")

    error_checks = [
        ("empty input", [], []),
        ("different lengths", [1, 2, 3], [1, 2]),
        ("not 1D input", [[1, 2], [3, 4]], [1, 2, 3, 4]),
    ]

    metric_functions = [
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        r2_score,
    ]

    for case_name, y_true, y_pred in error_checks:
        for metric_func in metric_functions:
            check_name = f"{metric_func.__name__} rejects {case_name}"
            try:
                metric_func(y_true, y_pred)
                passed = False
                print(f"[FAIL] {check_name}: did not raise ValueError")
            except ValueError:
                print(f"[PASS] {check_name}")
            except Exception as exc:
                passed = False
                print(f"[FAIL] {check_name}: raised {type(exc).__name__}: {exc}")

    return passed


if __name__ == "__main__":
    if not validate_metrics():
        raise SystemExit(1)
