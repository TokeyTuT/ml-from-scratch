import numpy as np
import pytest

from model import KNNClassifier, KNNRegression


@pytest.mark.parametrize("model_cls", [KNNClassifier, KNNRegression])
@pytest.mark.parametrize("n_neighbors", [0, -1, 1.5, True])
def test_invalid_n_neighbors_raises(model_cls, n_neighbors):
    with pytest.raises(ValueError, match="positive integer"):
        model_cls(n_neighbors=n_neighbors)


def test_k_larger_than_training_samples_raises():
    model = KNNClassifier(n_neighbors=3)

    with pytest.raises(ValueError, match="greater than the number of training samples"):
        model.fit([[0], [1]], [0, 1])


def test_predict_before_fit_raises():
    model = KNNClassifier()

    with pytest.raises(ValueError, match="Call fit before predict"):
        model.predict([[0]])


def test_regression_fit_converts_lists_to_arrays():
    model = KNNRegression(n_neighbors=2)
    model.fit([[0.0], [2.0]], [0.0, 2.0])

    prediction = model.predict([[1.0]])

    assert np.allclose(prediction, [1.0])


def test_regression_accepts_distance_metric_keyword():
    def custom_distance_metric(x_train, x):
        return np.array([1.0, 0.0])

    model = KNNRegression(n_neighbors=1, distance_metric=custom_distance_metric)
    model.fit([[0.0], [10.0]], [0.0, 10.0])

    prediction = model.predict([[5.0]])

    assert np.allclose(prediction, [10.0])
