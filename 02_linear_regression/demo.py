import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from linear_regression import LinearRegression
from utils import mean_squared_error


def format_array(values):
    """把数组压缩成适合命令行查看的短字符串。"""
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def print_result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")
    return passed


def run_check(name, check_func):
    try:
        passed, detail = check_func()
    except Exception as exc:
        detail = f"raised {type(exc).__name__}: {exc}"
        return print_result(name, False, detail)

    return print_result(name, passed, detail)


def expect_raises(name, action, expected_error=ValueError):
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


def check_single_feature_fit():
    X = np.linspace(-3.0, 3.0, 80).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    model = LinearRegression(
        learning_rate=0.05,
        epochs=5000,
        tolerance=1e-12,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = model.score(X, y)
    passed = (
        np.allclose(model.weights, np.array([2.0]), atol=1e-2)
        and np.isclose(model.bias, 1.0, atol=1e-2)
        and mse < 1e-4
        and r2 > 0.999
    )
    detail = (
        f"weights={format_array(model.weights)}, "
        f"bias={model.bias:.6f}, mse={mse:.8f}, r2={r2:.8f}"
    )
    return passed, detail


def check_multi_feature_fit():
    x1_values = np.linspace(-2.0, 2.0, 7)
    x2_values = np.array([-1.5, -0.5, 0.5, 1.5])
    X = np.array(
        [[x1, x2] for x1 in x1_values for x2 in x2_values],
        dtype=float,
    )
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 2.0

    model = LinearRegression(
        learning_rate=0.05,
        epochs=10000,
        tolerance=1e-12,
    )
    model.fit(X, y)

    passed = (
        np.allclose(model.weights, np.array([1.5, -0.5]), atol=1e-2)
        and np.isclose(model.bias, 2.0, atol=1e-2)
        and model.score(X, y) > 0.999
    )
    detail = (
        f"weights={format_array(model.weights)}, "
        f"bias={model.bias:.6f}, r2={model.score(X, y):.8f}"
    )
    return passed, detail


def check_noisy_data_stability():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 3))
    true_weights = np.array([1.2, -2.0, 0.5])
    true_bias = -0.75
    y = X @ true_weights + true_bias + rng.normal(scale=0.05, size=X.shape[0])

    model = LinearRegression(
        learning_rate=0.05,
        epochs=5000,
        tolerance=1e-12,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    finite_values = np.all(np.isfinite(y_pred))
    good_shape = y_pred.shape == y.shape
    good_score = model.score(X, y) > 0.98
    passed = finite_values and good_shape and good_score
    detail = (
        f"pred_shape={y_pred.shape}, "
        f"finite={finite_values}, r2={model.score(X, y):.8f}"
    )
    return passed, detail


def check_fit_without_intercept():
    X = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    y = -3.0 * X.ravel()

    model = LinearRegression(
        learning_rate=0.05,
        epochs=5000,
        tolerance=1e-12,
        fit_intercept=False,
    )
    model.fit(X, y)

    passed = (
        np.allclose(model.weights, np.array([-3.0]), atol=1e-2)
        and model.bias == 0
        and model.score(X, y) > 0.999
    )
    detail = (
        f"weights={format_array(model.weights)}, "
        f"bias={model.bias}, r2={model.score(X, y):.8f}"
    )
    return passed, detail


def check_refit_resets_parameters():
    model = LinearRegression(
        learning_rate=0.05,
        epochs=5000,
        tolerance=1e-12,
    )

    X_first = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    y_first = 4.0 * X_first.ravel() - 2.0
    model.fit(X_first, y_first)

    X_second = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y_second = -1.0 * X_second.ravel() + 3.0
    model.fit(X_second, y_second)

    passed = (
        np.allclose(model.weights, np.array([-1.0]), atol=1e-2)
        and np.isclose(model.bias, 3.0, atol=1e-2)
        and model.score(X_second, y_second) > 0.999
    )
    detail = (
        f"after_refit weights={format_array(model.weights)}, "
        f"bias={model.bias:.6f}"
    )
    return passed, detail


def main():
    results = [
        run_check("single feature exact line fit", check_single_feature_fit),
        run_check("multi feature exact plane fit", check_multi_feature_fit),
        run_check("stable fit on noisy data", check_noisy_data_stability),
        run_check("fit_intercept=False keeps bias at zero", check_fit_without_intercept),
        run_check("second fit resets learned parameters", check_refit_resets_parameters),
        expect_raises(
            "predict rejects unfitted model",
            lambda: LinearRegression().predict([[1.0]]),
        ),
        expect_raises(
            "fit rejects 1D X",
            lambda: LinearRegression().fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ),
        expect_raises(
            "fit rejects 2D y",
            lambda: LinearRegression().fit([[1.0], [2.0]], [[1.0], [2.0]]),
        ),
        expect_raises(
            "fit rejects mismatched sample counts",
            lambda: LinearRegression().fit([[1.0], [2.0]], [1.0]),
        ),
        expect_raises(
            "fit rejects empty training data",
            lambda: LinearRegression().fit(np.empty((0, 1)), np.array([])),
        ),
        expect_raises(
            "predict rejects wrong feature count",
            lambda: LinearRegression()
            .fit([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [1.0, 2.0, 3.0])
            .predict([[1.0, 2.0, 3.0]]),
        ),
        expect_raises(
            "predict rejects 1D X",
            lambda: LinearRegression()
            .fit([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [1.0, 2.0, 3.0])
            .predict([1.0, 2.0]),
        ),
        expect_raises(
            "fit rejects non-numeric X",
            lambda: LinearRegression().fit([["bad"], ["data"]], [1.0, 2.0]),
        ),
        expect_raises(
            "fit rejects NaN values",
            lambda: LinearRegression().fit([[1.0], [np.nan]], [1.0, 2.0]),
        ),
        expect_raises(
            "predict rejects infinite values",
            lambda: LinearRegression()
            .fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
            .predict([[np.inf]]),
        ),
        expect_raises(
            "score rejects mismatched sample counts",
            lambda: LinearRegression()
            .fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
            .score([[1.0], [2.0]], [1.0]),
        ),
    ]

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nSummary: {passed_count}/{total_count} checks passed.")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
