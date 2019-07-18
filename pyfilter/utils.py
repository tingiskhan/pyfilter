import numpy as np
from collections import Iterable
from .normalization import normalize
import torch
from torch.distributions import Distribution
from .timeseries.parameter import Parameter


def get_ess(w, normalized=False):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :type w: torch.Tensor
    :param normalized: Whether input is normalized
    :type normalized: bool
    :return: The effective sample size
    :rtype: torch.Tensor
    """

    if not normalized:
        w = normalize(w)

    return w.sum(-1) ** 2 / (w ** 2).sum(-1)


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

    return array[torch.arange(array.shape[0], device=array.device)[:, None], indices]


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

    # ===== Calculate the second term ===== #
    if weights is None:
        temp = torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw)).mean(-1).log()
    else:
        temp = (weights * torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw))).sum(-1).log()

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

    return x.view(*x.shape, *((ndim - x.dim()) * (1,)))


def concater(x):
    """
    Concatenates output.
    :type x: tuple[torch.Tensor]|torch.Tensor
    :rtype: torch.Tensor
    """

    if not isinstance(x, tuple):
        return x

    return torch.stack(torch.broadcast_tensors(*x), dim=-1)


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

    b = torch.eye(x.size(-1), device=x.device)
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


def _yield_helper(obj):
    for a in (d for d in dir(obj) if d != '__class__' and not d.startswith('__') and not d.endswith('__')):
        try:
            if isinstance(getattr(type(obj), a), property):
                continue
        except AttributeError:
            yield a


class MoveToHelper(object):
    _device = torch.empty([0]).device

    def _helper(self, device, attr):
        if isinstance(attr, Parameter):
            for a in _yield_helper(attr._prior):
                self._helper(device, getattr(attr._prior, a))

            attr.data = attr.data.to(device)
        elif hasattr(attr, 'to_'):
            attr.to_(device)
        elif isinstance(attr, torch.Tensor) and attr.device != device:
            attr.data = attr.data.to(device)
            return self
        elif isinstance(attr, (tuple, list)):
            for i in range(len(attr)):
                self._helper(device, attr[i])
        elif isinstance(attr, Distribution):
            for a in _yield_helper(attr):
                self._helper(device, getattr(attr, a))

        return self

    def to_(self, device):
        """
        Moves the current object to the specified device.
        :param device: The device to move to
        :type device: str
        :return: Self
        :rtype: MoveToHelper
        """

        self._device = torch.device(device)

        for a in _yield_helper(self):
            attr = getattr(self, a)
            self._helper(device, attr)

        return self