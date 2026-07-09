import numpy as np
from logistic_regression import LogisticRegression


def format_array(values):
    """把数组压缩成适合命令行查看的短字符串。"""
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def print_result(name, passed, detail=""):
    """打印单项检查结果。"""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")
    return passed


def run_check(name, check_func):
    """执行检查函数，并把异常转换成失败结果。"""
    try:
        passed, detail = check_func()
    except Exception as exc:
        detail = f"raised {type(exc).__name__}: {exc}"
        return print_result(name, False, detail)

    return print_result(name, passed, detail)


def expect_raises(name, action, expected_error=ValueError):
    """验证某个非法输入会抛出预期异常。"""
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


def make_binary_classification_data():
    """构造一个线性可分的二维二分类数据集。"""
    negative_samples = np.array(
        [
            [-3.0, -2.0],
            [-2.5, -1.0],
            [-2.0, -2.5],
            [-1.5, -1.0],
            [-1.0, -2.0],
            [-0.5, -1.5],
        ]
    )
    positive_samples = np.array(
        [
            [0.5, 1.5],
            [1.0, 2.0],
            [1.5, 1.0],
            [2.0, 2.5],
            [2.5, 1.5],
            [3.0, 2.0],
        ]
    )

    X = np.vstack([negative_samples, positive_samples])
    y = np.array([0] * len(negative_samples) + [1] * len(positive_samples))
    return X, y


def fit_demo_model():
    """训练一个用于多个检查复用的逻辑回归模型。"""
    X, y = make_binary_classification_data()
    model = LogisticRegression(
        learning_rate=0.2,
        epochs=3000,
        tolerance=1e-10,
    )
    model.fit(X, y)
    return model, X, y


def check_fit_and_score():
    model, X, y = fit_demo_model()

    y_pred = model.predict(X)
    score = model.score(X, y)
    passed = y_pred.shape == y.shape and score >= 0.95
    detail = (
        f"pred={format_array(y_pred)}, "
        f"score={score:.4f}, "
        f"weights={format_array(model.weights)}, "
        f"bias={model.bias:.4f}"
    )
    return passed, detail


def check_predict_proba_range():
    model, X, _ = fit_demo_model()

    probabilities = model.predict_proba(X)
    passed = probabilities.shape == (X.shape[0],) and np.all(
        (probabilities >= 0) & (probabilities <= 1)
    )
    detail = f"proba={format_array(probabilities)}"
    return passed, detail


def check_loss_decreases():
    model, _, _ = fit_demo_model()

    first_loss = model.loss_history[0]
    last_loss = model.loss_history[-1]
    passed = last_loss < first_loss
    detail = (
        f"first_loss={first_loss:.6f}, "
        f"last_loss={last_loss:.6f}, "
        f"steps={len(model.loss_history)}"
    )
    return passed, detail


def check_unseen_points():
    model, _, _ = fit_demo_model()

    X_test = np.array(
        [
            [-2.0, -1.5],
            [2.0, 1.5],
            [-1.0, -0.8],
            [1.0, 1.2],
        ]
    )
    expected = np.array([0, 1, 0, 1])
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    passed = np.array_equal(y_pred, expected)
    detail = (
        f"pred={format_array(y_pred)}, "
        f"expected={format_array(expected)}, "
        f"proba={format_array(probabilities)}"
    )
    return passed, detail


def main():
    results = [
        run_check("fit reaches high training accuracy", check_fit_and_score),
        run_check("predict_proba returns valid probabilities", check_predict_proba_range),
        run_check("training loss decreases", check_loss_decreases),
        run_check("predict handles unseen points", check_unseen_points),
        expect_raises(
            "predict rejects unfitted model",
            lambda: LogisticRegression().predict([[1.0, 2.0]]),
        ),
        expect_raises(
            "fit rejects labels outside 0/1",
            lambda: LogisticRegression().fit([[1.0], [2.0]], [0, 2]),
        ),
        expect_raises(
            "predict rejects wrong feature count",
            lambda: LogisticRegression()
            .fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])
            .predict([[1.0, 2.0, 3.0]]),
        ),
    ]

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
