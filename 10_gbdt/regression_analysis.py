"""使用合成非线性数据简单分析 GBDT 回归模型的效果。"""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gbdt import GBDTRegression
from utils import mean_absolute_error, mean_squared_error, r2_score


def make_regression_data(random_state=42):
    """生成带少量噪声的非线性回归数据。"""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-3.0, 3.0, size=(240, 2))
    noise = rng.normal(loc=0.0, scale=0.25, size=len(X))
    y = (
        2.5 * np.sin(X[:, 0])
        + 0.6 * X[:, 1] ** 2
        - 0.8 * X[:, 0] * X[:, 1]
        + noise
    )
    return X, y


def split_data(X, y, train_ratio=0.75, random_state=42):
    """把数据随机划分为训练集和测试集。"""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    split_index = int(len(X) * train_ratio)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def calculate_metrics(y_true, y_pred):
    """计算回归任务常用的三个评价指标。"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """训练模型并返回训练集、测试集上的评价结果。"""
    model.fit(X_train, y_train)
    train_metrics = calculate_metrics(y_train, model.predict(X_train))
    test_metrics = calculate_metrics(y_test, model.predict(X_test))

    return {
        "name": name,
        "train_mse": train_metrics["mse"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
    }


def print_results(results):
    """以表格形式输出各个模型的回归指标。"""
    header = (
        f"{'模型':<20}"
        f"{'训练 MSE':>12}"
        f"{'测试 MSE':>12}"
        f"{'测试 MAE':>12}"
        f"{'测试 R²':>12}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['name']:<20}"
            f"{result['train_mse']:>12.4f}"
            f"{result['test_mse']:>12.4f}"
            f"{result['test_mae']:>12.4f}"
            f"{result['test_r2']:>12.4f}"
        )


def main():
    X, y = make_regression_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline_train_pred = np.full(len(y_train), np.mean(y_train))
    baseline_test_pred = np.full(len(y_test), np.mean(y_train))
    baseline_train_metrics = calculate_metrics(y_train, baseline_train_pred)
    baseline_test_metrics = calculate_metrics(y_test, baseline_test_pred)

    results = [
        {
            "name": "均值基线",
            "train_mse": baseline_train_metrics["mse"],
            "test_mse": baseline_test_metrics["mse"],
            "test_mae": baseline_test_metrics["mae"],
            "test_r2": baseline_test_metrics["r2"],
        }
    ]

    for learning_rate in (0.1, 0.3, 1.0):
        model = GBDTRegression(
            n_estimators=30,
            learning_rate=learning_rate,
            max_depth=2,
        )
        results.append(
            evaluate_model(
                name=f"GBDT (lr={learning_rate})",
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )

    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}\n")
    print_results(results)

    baseline_mse = results[0]["test_mse"]
    best_result = min(results[1:], key=lambda result: result["test_mse"])
    mse_reduction = (baseline_mse - best_result["test_mse"]) / baseline_mse

    print("\n简单结论")
    print(
        f"- 当前表现最好的配置是 {best_result['name']}，"
        f"测试集 R² 为 {best_result['test_r2']:.4f}。"
    )
    print(f"- 与均值基线相比，测试集 MSE 降低了 {mse_reduction:.2%}。")
    print("- 训练 MSE 明显低于测试 MSE，说明模型存在一定的泛化差距。")


if __name__ == "__main__":
    main()
