import numpy as np

from project9417.residual_agop import residual_weighted_agop, standard_agop, top_eigvec


def test_standard_agop_matches_manual_average():
    grads = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = standard_agop(grads)

    expected = np.array([[5.0, 7.0], [7.0, 10.0]])
    np.testing.assert_allclose(result, expected)


def test_residual_weighted_agop_falls_back_when_weights_vanish():
    grads = np.array([[1.0, 0.0], [0.0, 1.0]])
    residuals = np.zeros(2)

    result = residual_weighted_agop(grads, residuals)

    np.testing.assert_allclose(result, standard_agop(grads))


def test_residual_weighted_agop_applies_squared_residual_weights():
    grads = np.array([[1.0, 0.0], [0.0, 2.0]])
    residuals = np.array([1.0, 3.0])

    result = residual_weighted_agop(grads, residuals)

    expected = np.array([[0.1, 0.0], [0.0, 3.6]])
    np.testing.assert_allclose(result, expected)


def test_top_eigvec_flips_sign_to_largest_magnitude_entry():
    matrix = np.array([[3.0, 0.0], [0.0, 1.0]])

    result = top_eigvec(matrix)

    np.testing.assert_allclose(result, np.array([1.0, 0.0]))
