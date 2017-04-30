import numpy as np


def _vector(weights):
    """
    Normalizes a 1D array of log weights.
    :param weights:
    :return:
    """
    n = weights.shape[0]

    max_log_weight = np.nanmax(weights)
    re_weighted = np.exp(weights - max_log_weight)
    sum_of_weights = np.nansum(re_weighted)

    normalized = re_weighted / sum_of_weights
    normalized[np.isnan(normalized)] = 0

    # ===== Remove Nans from normalized ===== #

    if sum(normalized) == 0:
        normalized = 1 / n * np.ones(n)

    return normalized


def _matrix(weights):
    """
    Normalizes a 2D array of log weights along the second axis.
    :param weights:
    :return:
    """
    n = weights.shape[1]

    max_weight = np.nanmax(weights, axis=1)

    altered_weights = np.exp(weights - max_weight[:, None])
    sum_of_weights = np.nansum(altered_weights, axis=1)[:, None]
    normalized = altered_weights / sum_of_weights
    normalized[np.isnan(normalized)] = 0

    # ===== Remove Nans from normalized ===== #

    uniform_probability = 1 / n * np.ones(n)
    zero_rows = np.where(np.sum(normalized, axis=1) == 0)
    normalized[zero_rows, :] = uniform_probability

    return normalized


def normalize(weights):
    """
    Normalizes a 1D or 2D array of log weights.
    :param weights:
    :return:
    """

    if weights.ndim > 1:
        return _matrix(weights)

    return _vector(weights)