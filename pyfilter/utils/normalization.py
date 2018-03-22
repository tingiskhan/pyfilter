import numpy as np


def _vector(w):
    """
    Normalizes a 1D array of log weights.
    :param w: The weights
    :type w: np.ndarray
    :return: Normalized weights
    :rtype: np.ndarray
    """

    reweighed = np.exp(w - w.max())

    normalized = reweighed / reweighed.sum()
    normalized[np.isnan(normalized)] = 0

    # ===== Remove Nans from normalized ===== #

    if sum(normalized) == 0:
        n = w.shape[0]
        normalized = np.ones(n) / n

    return normalized


def _matrix(w):
    """
    Normalizes a 2D array of log weights along the second axis.
    :param w: The weights
    :type w: np.ndarray
    :return: Normalized weights
    :rtype: np.ndarray
    """

    reweighed = np.exp(w - w.max(axis=-1)[..., None])
    normalized = reweighed / reweighed.sum(axis=-1)[..., None]
    normalized[np.isnan(normalized)] = 0

    # ===== Remove Nans from normalized ===== #

    mask = normalized.sum(axis=-1) == 0
    n = w.shape[-1]
    normalized[mask] = np.ones(n) / n

    return normalized


def normalize(w):
    """
    Normalizes a 1D or 2D array of log weights.
    :param w: The weights
    :type w: np.ndarray
    :return: Normalized weights
    :rtype: np.ndarray
    """

    if w.ndim > 1:
        return _matrix(w)

    return _vector(w)