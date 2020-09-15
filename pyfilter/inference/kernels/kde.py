import torch
from ...utils import get_ess
from math import sqrt
from typing import Union
from ..utils import _construct_mvn
from torch.distributions import Normal, Independent


def _jitter(values: torch.Tensor, scale: Union[float, torch.Tensor]):
    """
    Jitters the parameters.
    :param values: The values
    :param scale: The scaling to use for the variance of the proposal
    :return: Proposed values
    """

    return values + scale * torch.empty_like(values).normal_()


def silverman(n: int, ess: float):
    """
    Returns Silverman's factor.
    :param n: The dimension
    :param ess: The ess
    :return: Bandwidth factor
    """

    return (ess * (n + 2) / 4) ** (-1 / (n + 4))


def scott(n: int, ess: float):
    """
    Returns Silverman's factor.
    :param n: The dimension
    :param ess: The ess
    :return: Bandwidth factor
    """

    return 1.059 * ess ** (-1 / (n + 4))


def robust_var(x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor = None):
    """
    Calculates the scale robustly
    :param x: The values
    :param w: The weights
    :param mean: The mean
    """

    # ===== IQR ===== #
    sort, inds = x.sort(0)
    cumw = w[inds].cumsum(0)

    lowind = (cumw - 0.25).abs().argmin(0)
    highind = (cumw - 0.75).abs().argmin(0)

    iqr = (sort[highind].diag() - sort[lowind].diag()) / 1.349
    iqr2 = iqr ** 2

    # ===== Calculate std regularly ===== #
    w = w.unsqueeze(-1)

    if mean is None:
        mean = (w * x).sum(0)

    var = (w * (x - mean) ** 2).sum(0)

    # ===== Check which to use ===== #
    mask = iqr2 <= var

    if mask.any():
        var[mask] = iqr2[mask]

    return var


class KernelDensityEstimate(object):
    def __init__(self):
        """
        Implements the base class for KDEs.
        """

        self._cov = None
        self._bw_fac = None
        self._means = None

    def fit(self, x: torch.Tensor, w: torch.Tensor):
        """
        Fits the KDE.
        :param x: The samples
        :param w: The weights
        :return: Self
        """

        raise NotImplementedError()

    def sample(self, inds: torch.Tensor = None):
        """
        Samples from the KDE.
        :param inds: Whether to manually specify the samples chosen
        :return: New samples
        """

        raise NotImplementedError()

    def get_ess(self, w):
        return get_ess(w, normalized=True)


class ShrinkingKernel(KernelDensityEstimate):
    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = self.get_ess(w)
        self._bw_fac = 1.59 * ess ** (-1 / 3)

        # ===== Calculate variance ===== #
        mean = (w.unsqueeze(-1) * x).sum(0)
        self._cov = robust_var(x, w, mean)

        # ===== Calculate shrinkage and shrink ===== #
        beta = sqrt(1. - self._bw_fac ** 2)
        self._means = mean + beta * (x - mean)

        return self

    def sample(self, inds=None):
        inds = inds if inds is not None else torch.arange(self._means.shape[0], device=self._means.device)
        return _jitter(self._means[inds], self._bw_fac * self._cov.sqrt())


class NonShrinkingKernel(ShrinkingKernel):
    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = self.get_ess(w)
        self._bw_fac = 1.59 * ess ** (-1 / 3)

        # ===== Calculate variance ===== #
        self._cov = robust_var(x, w)

        # ===== Calculate shrinkage and shrink ===== #
        self._means = x

        return self


class LiuWestShrinkage(ShrinkingKernel):
    def __init__(self, delta=0.99):
        super().__init__()
        self._a = delta
        self._bw_fac = sqrt(1 - delta ** 2)

    def fit(self, x, w):
        mean = (w.unsqueeze(-1) * x).sum(0)

        self._cov = robust_var(x, w, mean)
        self._means = x * self._a + (1 - self._a) * mean

        return self


class IndependentGaussian(ShrinkingKernel):
    def __init__(self, factor=silverman):
        """
        Implements a Gaussian KDE.
        :param factor: How to calculate the factor bandwidth
        """
        super().__init__()
        self._fac = factor
        self._w = None

    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = self.get_ess(w)
        self._bw_fac = self._fac(x.shape[-1], ess)

        # ===== Calculate covariance ===== #
        self._cov = robust_var(x, w)
        self._w = w
        self._means = x

        return self


class ConstantKernel(ShrinkingKernel):
    def __init__(self, bw: Union[float, torch.Tensor]):
        """
        Kernel with constant, prespecified bandwidth.
        :param bw: The bandwidth to use
        """
        super().__init__()
        self._bw_fac = bw
        self._w = None

    def fit(self, x, w):
        self._w = w
        self._means = x
        self._cov = torch.ones(self._means.shape[-1], device=self._means.device)

        return self


# TODO: Not really a KDE...
class NormalApproximation(KernelDensityEstimate):
    def __init__(self, independent=True):
        super().__init__()
        self._dist = None   # type: torch.distributions.Distribution
        self._indep = independent
        self._shape = None

    def fit(self, x, w):
        self._shape = (x.shape[0],)

        if not self._indep:
            self._dist = _construct_mvn(x, w)
            return self

        mean = (w.unsqueeze(-1) * x).sum(0)
        var = robust_var(x, w, mean)

        self._dist = Independent(Normal(mean, var.sqrt()), 1)

        return self

    def sample(self, inds=None):
        return self._dist.sample(self._shape)
