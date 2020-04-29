import numpy as np
from collections import Iterable
from .normalization import normalize
import torch
from torch.distributions import Distribution
import numbers
from math import sqrt
from scipy.stats import chi2


_NATIVE = (bool, str, numbers.Number)
EPS = sqrt(torch.finfo(torch.float32).eps)


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
    r = max_num * torch.arange(a.shape[0], dtype=max_num.dtype, device=a.device)[:, None]
    p = torch.bucketize((a + r).view(-1), (b + r).view(-1)).reshape(m, -1)
    return p - n * torch.arange(m, dtype=p.dtype, device=a.device)[:, None]


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
        temp = (
            torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw))
            .mean(-1)
            .log()
        )
    else:
        temp = (
            (weights * torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw)))
            .sum(-1)
            .log()
        )

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


def concater(*x):
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


def flatten(*args):
    """
    Flattens an array comprised of an arbitrary number of lists. Solution found at:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    :param args: The iterable you wish to flatten.
    :type args: collections.Iterable
    :return:
    """
    out = list()
    for el in args:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, torch.Tensor)):
            out.extend(flatten(*el))
        else:
            out.append(el)

    return tuple(out)


def normal_test(x, alpha=0.05):
    """
    Implements a basic Jarque-Bera test for normality.
    :param x: The data
    :type x: torch.Tensor
    :param alpha: The level of confidence
    :type alpha: float
    :return: Whether a normal distribution or not
    :rtype: bool
    """
    mean = x.mean(0)
    var = ((x - mean) ** 2).mean(0)

    # ===== Skew ===== #
    skew = ((x - mean) ** 3).mean(0) / var ** 1.5

    # ===== Kurtosis ===== #
    kurt = ((x - mean) ** 4).mean(0) / var ** 2

    # ===== Statistic ===== #
    jb = x.shape[0] / 6 * (skew ** 2 + 1 / 4 * (kurt - 3) ** 2)

    return chi2(2).ppf(1 - alpha) >= jb


def unflattify(values, shape):
    """
    Unflattifies parameter values.
    :param values: The flattened array of values that are to be unflattified
    :type values: torch.Tensor
    :param shape: The shape of the parameter prior
    :type shape: torch.Size
    :rtype: torch.Tensor
    """

    if len(shape) < 1 or values.shape[1:] == shape:
        return values

    return values.reshape(values.shape[0], *shape)


class TempOverride(object):
    def __init__(self, obj, attr, new_vals):
        """
        Implements a temporary override of attribute of an object.
        :param obj: An object
        :type obj: object
        :param attr: The attribute to override
        :type attr: str
        :param new_vals: The new values
        :type new_vals: object
        """
        self._obj = obj
        self._attr = attr
        self._new_vals = new_vals
        self._old_vals = None

    def __enter__(self):
        self._old_vals = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, self._new_vals)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self._obj, self._attr, self._old_vals)

        return False


class Empirical(Distribution):
    def __init__(self, samples):
        """
        Helper class for timeseries without an analytical expression.
        :param samples: The sample
        :type samples: torch.Tensor
        """
        super().__init__()
        self.loc = self._samples = samples
        self.scale = torch.zeros_like(samples)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape != self._samples.shape and sample_shape != torch.Size():
            raise ValueError('Current implementation only allows passing an empty size!')

        return self._samples