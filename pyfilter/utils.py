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
    :rtype: float
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


def loglikelihood(w):
    """
    Calculates the estimated loglikehood given weights.
    :param w: The log weights, corresponding to likelihood
    :type w: torch.Tensor
    :return: The log-likelihood
    :rtype: torch.Tensor
    """

    maxw, _ = w.max(-1)

    axis = -1
    if maxw.dim() > 0:
        axis = 0
        w = w.t()

    return maxw + torch.log(torch.exp(w - maxw).mean(axis))


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


def broadcast_all(*args):
    """
    Basically same as Torch's, but on the other axis.
    :type args: tuple[torch.Tensor]
    :rtype: tuple[torch.Tensor]
    """
    # TODO: Switch to PyTorch's when it comes

    biggest = max(a.dim() for a in args)

    out = tuple()
    for a in args:
        out += (add_dimensions(a, biggest),)

    return out


def isfinite(x):
    """
    Returns mask for finite values. Solution: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519 .
    :param x: The array
    :type x: torch.Tensor
    :return: All those that do not satisfy
    :rtype: torch.Tensor
    """

    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return not_inf & not_nan


def concater(x):
    """
    Concatenates output.
    :type x: tuple[torch.Tensor]|torch.Tensor
    :rtype: torch.Tensor
    """

    if not isinstance(x, tuple):
        return x

    return torch.cat(tuple(tx.unsqueeze(-1) for tx in x), dim=-1)


def construct_diag(x):
    """
    Constructs a diagonal matrix based on batched data. Solution found here:
    https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    :param x: The tensor
    :type x: torch.Tensor
    :rtype: torch.Tensor
    """

    if x.dim() < 1 or x.shape[-1] < 2:
        return x
    elif x.dim() < 2:
        return torch.diag(x)

    b = torch.eye(x.size(-1))
    c = x.unsqueeze(-1).expand(*x.size(), x.size(-1))

    return c * b


def approx_fprime(x, f, epsilon):
    """
    Wrapper for scipy's `approx_fprime`. Handles vectorized functions.
    :param x: The point at which to approximate the gradient
    :type x: torch.Tensor
    :param f: The function to approximate
    :type f: callable
    :param epsilon: The discretization to use
    :type epsilon: float
    :return: The gradient
    :rtype: torch.Tensor
    """

    f0 = f(x)

    grad = np.zeros_like(x)
    ei = np.zeros_like(x)

    for k in range(x.shape[0]):
        ei[k] = 1.
        d = epsilon * ei
        grad[k] = (f(x + d) - f0) / d[k]
        ei[k] = 0.

    return grad


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

