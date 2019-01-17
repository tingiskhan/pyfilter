import numpy as np
from collections import Iterable
from .normalization import normalize
import torch


def get_ess(w):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :type w: torch.Tensor
    :return: The effective sample size
    :rtype: torch.Tensor
    """

    normalized = normalize(w)

    return normalized.sum(-1) ** 2 / (normalized ** 2).sum(-1)


def searchsorted2d(a, b):
    """
    Searches a sorted 2D array along the second axis. Basically performs a vectorized digitize. Solution provided here:
    https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy.
    :param a: The array to take the elements from
    :type a: torch.Tensor
    :param b: The indices of the elements in `a`.
    :type b: torch.Tensor
    :return: An array of indices
    :rtype: torch.Tensor
    """
    m, n = a.shape
    max_num = max(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * torch.arange(a.shape[0], dtype=max_num.dtype)[:, None]
    p = np.searchsorted((a + r).view(-1), (b + r).view(-1)).reshape(m, -1)
    return p - n * torch.arange(m, dtype=p.dtype)[:, None]


def choose(array, indices):
    """
    Function for choosing on either columns or index.
    :param array: The array to choose on
    :type array: torch.Tensor
    :param indices: The indices to choose from `array`
    :type indices: torch.Tensor
    :return: Returns the chosen elements from `array`
    :rtype: torch.Tensor
    """

    if indices.dim() < 2:
        return array[indices]

    return array[torch.arange(array.shape[0])[:, None], indices]


def loglikelihood(w, weights=None):
    """
    Calculates the estimated loglikehood given weights.
    :param w: The log weights, corresponding to likelihood
    :type w: torch.Tensor
    :param weights: Whether to weight the log-likelihood.
    :type weights: torch.Tensor
    :return: The log-likelihood
    :rtype: torch.Tensor
    """

    maxw, _ = w.max(-1)

    axis = -1
    maxw = maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw

    # ===== Calculate the second term ===== #
    if weights is None:
        temp = torch.exp(w - maxw).mean(axis).log()
    else:
        temp = (weights * torch.exp(w - maxw)).sum(axis).log()

    return maxw + temp


def add_dimensions(x, ndim):
    """
    Adds the required dimensions to `x`.
    :param x: The array
    :type x: torch.Tensor
    :param ndim: The dimension
    :type ndim: int
    :rtype: torch.Tensor
    """

    if not isinstance(x, torch.Tensor):
        return x

    for i in range(ndim - x.dim()):
        x = x.unsqueeze(-1)

    return x


def isfinite(x):
    """
    Returns mask for finite values. Solution: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519 .
    :param x: The array
    :type x: torch.Tensor
    :return: All those that do not satisfy
    :rtype: torch.Tensor
    """

    not_inf = (x + 1) != x
    not_nan = x == x
    return not_inf & not_nan


def concater(x):
    """
    Concatenates output.
    :type x: tuple[torch.Tensor]|torch.Tensor
    :rtype: torch.Tensor
    """

    if not isinstance(x, tuple):
        return x

    return torch.cat(tuple(tx.unsqueeze(-1) for tx in torch.broadcast_tensors(*x)), dim=-1)


def construct_diag(x):
    """
    Constructs a diagonal matrix based on batched data. Solution found here:
    https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    Do note that it only considers the last axis.
    :param x: The tensor
    :type x: torch.Tensor
    :rtype: torch.Tensor
    """

    if x.dim() < 1:
        return x
    elif x.shape[-1] < 2:
        return x.unsqueeze(-1)
    elif x.dim() < 2:
        return torch.diag(x)

    b = torch.eye(x.size(-1))
    c = x.unsqueeze(-1).expand(*x.size(), x.size(-1))

    return c * b


def flatten(iterable):
    """
    Flattens an array comprised of an arbitrary number of lists. Solution found at:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    :param iterable: The iterable you wish to flatten.
    :type iterable: collections.Iterable
    :return:
    """
    out = list()
    for el in iterable:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, torch.Tensor)):
            out.extend(flatten(el))
        else:
            out.append(el)

    return out

