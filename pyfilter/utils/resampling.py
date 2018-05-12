import numpy as np
from .normalization import normalize
from ..utils.utils import searchsorted2d


def _matrix(weights):
    """
    Performs systematic resampling of a 2D array of log weights along the second axis.
    independent of the others.
    :param weights: The weights to use for resampling
    :type weights: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """
    n = weights.shape[1]
    u = np.random.uniform(size=weights.shape[0])[:, None]
    index_range = np.arange(n)[None, :] * np.ones(weights.shape)

    probs = (index_range + u) / n

    normalized = normalize(weights)
    cumsum = np.zeros((weights.shape[0], n + 1))
    cumsum[:, 1:] = normalized.cumsum(axis=1)

    return searchsorted2d(cumsum, probs).astype(int) - 1


def _vector(weights):
    """
    Performs systematic resampling of a 1D array log weights.
    :param weights: The weights to use for resampling
    :type weights: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """
    n = weights.size
    u = np.random.uniform()
    probs = (np.arange(n) + u) / n

    cumsum = np.zeros(n + 1)
    normalized = normalize(weights)
    cumsum[1:] = normalized.cumsum()

    return np.searchsorted(cumsum, probs).astype(int) - 1


def systematic(w):
    """
    Performs systematic resampling on either a 1D or 2D array.
    :param w: The weights to use for resampling
    :type w: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """
    if w.ndim > 1:
        return _matrix(w)

    return _vector(w)


def _mn_vector(w):
    """
    Resamples a vector array of weights using multinomial resampling.
    :param w: The weights to use for resampling
    :type w: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """
    normalized = normalize(w).cumsum()
    normalized[-1] = 1

    return np.searchsorted(normalized, np.random.uniform(0, 1, w.shape))


def _mn_matrix(w):
    """
    Resamples a matrix array of weights using multinomial resampling.
    :param w: The weights to use for resampling
    :type w: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """

    normalized = normalize(w).cumsum(axis=-1)
    normalized[:, -1] = 1

    return searchsorted2d(normalized, np.random.uniform(0, 1, w.shape))


def multinomial(w):
    """
    Performs multinomial resampling on either a 1D or 2D array.
    :param w: The weights to use for resampling
    :type w: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """

    if w.ndim > 1:
        return _mn_matrix(w)

    return _mn_vector(w)
