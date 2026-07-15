"""DecisionTreeClassifier 的直接运行演示与行为检查。"""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_tree_classifier import DecisionTreeClassifier


def format_array(values):
    """把数组压缩成适合命令行查看的字符串。"""
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def print_result(name, passed, detail=""):
    """打印单项检查结果。"""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")
    return passed


def run_check(name, check_func):
    """运行检查函数，并把意外异常转换为失败结果。"""
    try:
        passed, detail = check_func()
    except Exception as exc:
        detail = f"raised {type(exc).__name__}: {exc}"
        return print_result(name, False, detail)

    return print_result(name, passed, detail)


def expect_raises(name, action, expected_error=ValueError):
    """检查非法输入是否抛出预期异常。"""
    try:
        action()
    except expected_error as exc:
        detail = f"raised {type(exc).__name__}: {exc}"
        return print_result(name, True, detail)
    except Exception as exc:
        detail = (
            f"raised {type(exc).__name__}, "
            f"expected {expected_error.__name__}: {exc}"
        )
        return print_result(name, False, detail)

    return print_result(name, False, f"did not raise {expected_error.__name__}")


def make_classification_data():
    """构造一个能按第 0 个特征清晰分开的二维数据集。"""
    X = np.array(
        [
            [1.0, 1.2],
            [1.3, 2.0],
            [1.8, 0.8],
            [3.0, 1.0],
            [3.4, 2.1],
            [4.0, 0.7],
        ]
    )
    y = np.array(["低风险", "低风险", "低风险", "高风险", "高风险", "高风险"])
    return X, y


def fit_demo_model():
    """训练一个供多个检查复用的决策树。"""
    X, y = make_classification_data()
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    return model, X, y


def check_gini_impurity():
    """检查 Gini 只由类别比例决定，不受标签具体数值影响。"""
    model = DecisionTreeClassifier()
    binary_gini = model._impurity_measure([0, 0, 1, 1])
    relabeled_gini = model._impurity_measure([10, 10, 20, 20])
    pure_gini = model._impurity_measure(["A", "A", "A"])

    passed = (
        np.isclose(binary_gini, 0.5)
        and np.isclose(relabeled_gini, 0.5)
        and np.isclose(pure_gini, 0.0)
    )
    detail = (
        f"balanced={binary_gini:.4f}, "
        f"relabeled={relabeled_gini:.4f}, pure={pure_gini:.4f}"
    )
    return passed, detail


def check_fit_and_training_prediction():
    """检查模型能找到正确根划分并拟合训练数据。"""
    model, X, y = fit_demo_model()
    y_pred = model.predict(X)

    passed = (
        model._root.feature_index == 0
        and np.isclose(model._root.threshold, 2.4)
        and y_pred.shape == y.shape
        and np.array_equal(y_pred, y)
    )
    detail = (
        f"feature={model._root.feature_index}, "
        f"threshold={model._root.threshold:.4f}, "
        f"pred={format_array(y_pred)}"
    )
    return passed, detail


def check_unseen_samples():
    """检查训练集之外的样本会沿着正确路径预测。"""
    model, _, _ = fit_demo_model()
    X_test = np.array([[1.5, 9.0], [3.2, -5.0]])
    expected = np.array(["低风险", "高风险"])
    y_pred = model.predict(X_test)

    passed = np.array_equal(y_pred, expected)
    detail = f"pred={format_array(y_pred)}, expected={format_array(expected)}"
    return passed, detail


def check_numeric_string_features():
    """检查可以转换为浮点数的数字字符串特征。"""
    X = [["0.0"], ["1.0"], ["2.0"], ["3.0"]]
    y = ["左侧", "左侧", "右侧", "右侧"]
    model = DecisionTreeClassifier().fit(X, y)
    y_pred = model.predict([["0.5"], ["2.5"]])
    expected = np.array(["左侧", "右侧"])

    passed = np.array_equal(y_pred, expected)
    detail = f"pred={format_array(y_pred)}"
    return passed, detail


def check_pure_node():
    """检查单类别数据直接生成叶子节点。"""
    model = DecisionTreeClassifier().fit(
        [[0.0], [1.0], [2.0]],
        ["同一类", "同一类", "同一类"],
    )
    y_pred = model.predict([[-10.0], [10.0]])
    root_is_leaf = model._root.left is None and model._root.right is None

    passed = root_is_leaf and np.array_equal(y_pred, ["同一类", "同一类"])
    detail = f"root_is_leaf={root_is_leaf}, pred={format_array(y_pred)}"
    return passed, detail


def check_max_depth_zero():
    """检查 max_depth=0 时根节点作为多数类叶子。"""
    model = DecisionTreeClassifier(max_depth=0).fit(
        [[0.0], [1.0], [2.0], [3.0]],
        [0, 0, 0, 1],
    )
    y_pred = model.predict([[0.0], [3.0]])
    root_is_leaf = model._root.left is None and model._root.right is None

    passed = root_is_leaf and np.array_equal(y_pred, [0, 0])
    detail = f"root_is_leaf={root_is_leaf}, pred={format_array(y_pred)}"
    return passed, detail


def collect_leaf_sizes(node):
    """递归收集树中所有叶子节点的样本数量。"""
    if node.left is None and node.right is None:
        return [node.n_samples]
    return collect_leaf_sizes(node.left) + collect_leaf_sizes(node.right)


def check_min_samples_leaf():
    """检查每个叶子的样本数都满足 min_samples_leaf。"""
    model = DecisionTreeClassifier(min_samples_leaf=2).fit(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        [0, 0, 1, 1, 1, 1],
    )
    leaf_sizes = collect_leaf_sizes(model._root)

    passed = all(size >= 2 for size in leaf_sizes)
    detail = f"leaf_sizes={leaf_sizes}"
    return passed, detail


def main():
    results = [
        run_check("Gini impurity is correct", check_gini_impurity),
        run_check("fit learns the expected split", check_fit_and_training_prediction),
        run_check("predict handles unseen samples", check_unseen_samples),
        run_check("numeric string features are accepted", check_numeric_string_features),
        run_check("pure labels create a leaf", check_pure_node),
        run_check("max_depth=0 creates a root leaf", check_max_depth_zero),
        run_check("min_samples_leaf is respected", check_min_samples_leaf),
        expect_raises(
            "predict rejects an unfitted model",
            lambda: DecisionTreeClassifier().predict([[1.0]]),
        ),
        expect_raises(
            "fit rejects 1D X",
            lambda: DecisionTreeClassifier().fit([1.0, 2.0], [0, 1]),
        ),
        expect_raises(
            "fit rejects mismatched sample counts",
            lambda: DecisionTreeClassifier().fit([[1.0], [2.0]], [0]),
        ),
        expect_raises(
            "predict rejects the wrong feature count",
            lambda: fit_demo_model()[0].predict([[1.0, 2.0, 3.0]]),
        ),
        expect_raises(
            "fit rejects non-numeric categorical features",
            lambda: DecisionTreeClassifier().fit(
                [["红色"], ["蓝色"]],
                [0, 1],
            ),
        ),
        expect_raises(
            "fit rejects NaN features",
            lambda: DecisionTreeClassifier().fit([[1.0], [np.nan]], [0, 1]),
        ),
        expect_raises(
            "constructor rejects invalid min_samples_split",
            lambda: DecisionTreeClassifier(min_samples_split=1),
        ),
    ]

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nSummary: {passed_count}/{total_count} checks passed.")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
