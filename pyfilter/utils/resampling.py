import numpy as np
from .normalization import normalize


def _matrix(weights):
    """
    Performs systematic resampling of a 2D array of log weights along the second axis.
    independent of the others.
    :param weights:
    :rtype: np.ndarray
    """
    n = weights.shape[1]
    u = np.random.uniform(size=weights.shape[0])[:, None]
    index_range = np.tile(np.arange(n), weights.shape[0]).reshape(weights.shape)

    probs = (index_range + u) / n

    normalized = normalize(weights)
    cumsum = np.zeros((weights.shape[0], n + 1))
    cumsum[:, 1:] = normalized.cumsum(axis=1)

    indices = np.empty_like(weights, dtype=int)
    for i in range(weights.shape[0]):
        indices[i, :] = np.digitize(probs[i, :], cumsum[i, :]) - 1

    # cumsum = normalized.cumsum(axis=1)
    # indices = _searchsorted2d(probs, cumsum).astype(int) - 1

    return indices


def _vector(weights):
    """
    Performs systematic resampling of a 1D array log weights.
    :param weights: The weights
    :type weights: np.ndarray
    :rtype: np.ndarray
    """
    n = weights.size
    u = np.random.uniform()
    probs = (np.arange(n) + u) / n

    cumsum = np.zeros(n + 1)
    normalized = normalize(weights)
    cumsum[1:] = normalized.cumsum()

    return np.digitize(probs, cumsum).astype(int) - 1


def systematic(w):
    """
    Performs systematic resampling on either a 1D or 2D array.
    :param w: The weights
    :type w: np.ndarray
    :return:
    """
    if w.ndim > 1:
        return _matrix(w)

    return _vector(w)


def _mn_vector(weights):
    """
    Resamples a vector array of weights using multinomial resampling.
    :param weights:
    :return:
    """
    normalized = normalize(weights)

    return np.random.choice(weights.size, weights.size, p=normalized)


def _mn_matrix(weights):
    """
    Resamples a matrix array of weights using multinomial resampling.
    :param weights: The weights
    :type weights: np.ndarray
    :return:
    """

    out = np.empty_like(weights, dtype=int)

    for i in range(weights.shape[0]):
        out[i] = _mn_vector(weights[i])

    return out


def multinomial(w):
    """
    Performs multinomial resampling on either a 1D or 2D array.
    :param w:
    :return:
    """

    if w.ndim > 1:
        return _mn_matrix(w)

    return _mn_vector(w)
