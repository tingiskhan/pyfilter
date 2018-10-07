import numpy as np
from .normalization import normalize
from ..utils.utils import searchsorted2d
import torch


def _matrix(weights, u):
    """
    Performs systematic resampling of a 2D array of log weights along the second axis.
    independent of the others.
    :param weights: The weights to use for resampling
    :type weights: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """
    n = weights.shape[1]
    u = torch.tensor(u) if u is not None else torch.empty((weights.shape[0], 1)).uniform_()
    index_range = torch.arange(n, dtype=u.dtype)[None, :] * torch.ones(weights.shape, dtype=u.dtype)

    probs = (index_range + u) / n

    cumsum = normalize(weights).cumsum(-1)

    return searchsorted2d(cumsum, probs)


def _vector(weights, u):
    """
    Performs systematic resampling of a 1D array log weights.
    :param weights: The weights to use for resampling
    :type weights: np.ndarray
    :return: Resampled indices
    :rtype: np.ndarray
    """
    n = weights.shape[0]
    u = torch.tensor(u) if u is not None else torch.empty(1).uniform_()
    probs = (torch.arange(n, dtype=u.dtype) + u) / n

    cumsum = normalize(weights).cumsum(0)

    return np.searchsorted(cumsum, probs)


def systematic(w, u=None):
    """
    Performs systematic resampling on either a 1D or 2D array.
    :param w: The weights to use for resampling
    :type w: np.ndarray
    :param u: Parameter for overriding the sampled index, for testing
    :type u: sample from np.random.uniform()
    :return: Resampled indices
    :rtype: np.ndarray
    """

    if w.dim() > 1:
        return _matrix(w, u)

    return _vector(w, u)


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

    normalized = normalize(w).cumsum(-1)
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
