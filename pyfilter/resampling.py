import numpy as np
from .normalization import normalize
from .utils import searchsorted2d
import torch


def _matrix(weights, u):
    """
    Performs systematic resampling of a 2D array of log weights along the second axis.
    independent of the others.
    :param weights: The weights to use for resampling
    :type weights: torch.Tensor
    :return: Resampled indices
    :rtype: torch.Tensor
    """
    n = weights.shape[1]
    index_range = torch.arange(n, dtype=u.dtype)[None, :] * torch.ones(weights.shape, dtype=u.dtype)

    probs = (index_range + u) / n
    cumsum = weights.cumsum(-1)

    return searchsorted2d(cumsum, probs)


def _vector(weights, u):
    """
    Performs systematic resampling of a 1D array log weights.
    :param weights: The weights to use for resampling
    :type weights: torch.Tensor
    :return: Resampled indices
    :rtype: torch.Tensor
    """
    n = weights.shape[0]
    probs = (torch.arange(n, dtype=u.dtype) + u) / n

    cumsum = weights.cumsum(0)

    return np.searchsorted(cumsum, probs)


def systematic(w, u=None):
    """
    Performs systematic resampling on either a 1D or 2D array.
    :param w: The weights to use for resampling
    :type w: torch.Tensor
    :param u: Parameter for overriding the sampled index, for testing
    :type u: float|torch.Tensor
    :return: Resampled indices
    :rtype: torch.Tensor
    """

    u = u if u is not None else (torch.empty(1) if w.dim() < 2 else torch.empty((w.shape[0], 1))).uniform_()
    w = normalize(w)

    if w.dim() > 1:
        return _matrix(w, u)

    return _vector(w, u)


def multinomial(w):
    """
    Performs multinomial sampling.
    :param w: The weights to use for resampling
    :type w: torch.Tensor
    :return: Resampled indices
    :rtype: torch.Tensor
    """

    return torch.multinomial(normalize(w), w.shape[-1], replacement=True)


def residual(w):
    """
    Performs residual resampling on a 1D array. Inspired by solution provided by the package "particles" on GitHub
    authored by the user "nchopin".
    :param w: The weights to use for resampling
    :type w: torch.Tensor
    :return: Resampled indices
    :rtype: torch.Tensor
    """

    w = normalize(w)

    if w.dim() > 1:
        raise NotImplementedError('Currently only 1-dimensional arrays supported!')

    floored = (w * w.shape[0]).floor().long()
    inds = torch.ones_like(w, dtype=floored.dtype)

    repeated = (torch.arange(w.shape[0], dtype=inds.dtype) * inds).repeat_interleave(floored)

    numtosample = w.shape[0] - floored.sum(0)

    inds[:-numtosample] = repeated
    inds[-numtosample:] = multinomial(torch.zeros(numtosample))

    return inds