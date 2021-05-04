import torch
from math import sqrt
from typing import Union
from ....utils import get_ess
from ....constants import EPS, INFTY


def _jitter(values: torch.Tensor, scale: Union[float, torch.Tensor]) -> torch.Tensor:
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

    sort, sort_indices = x.sort(0)
    cumulative_weights = w[sort_indices].cumsum(0)

    low_indices = (cumulative_weights - 0.25).abs().argmin(0)
    high_indices = (cumulative_weights - 0.75).abs().argmin(0)

    iqr = (sort[high_indices].diag() - sort[low_indices].diag()) / 1.349
    iqr2 = iqr ** 2

    w = w.unsqueeze(-1)

    if mean is None:
        mean = (w * x).sum(0)

    var = (w * (x - mean) ** 2).sum(0)

    mask = iqr2 <= var

    if mask.any():
        var[mask] = iqr2[mask]

    return var


class KernelDensityEstimate(object):
    """
    Implements the base class for KDEs.
    """

    def __init__(self, lowest_std: float = EPS):
        self._cov = None
        self._bw_fac = None
        self._means = None
        self._lowest_std = lowest_std

    def fit(self, x: torch.Tensor, w: torch.Tensor) -> "KernelDensityEstimate":
        """
        Fits the KDE.
        :param x: The samples
        :param w: The weights
        """

        raise NotImplementedError()

    def sample(self, indices: torch.Tensor = None) -> torch.Tensor:
        """
        Samples from the KDE.
        :param indices: Whether to manually specify the samples chosen
        """

        indices = indices if indices is not None else torch.arange(self._means.shape[0], device=self._means.device)
        std = (self._bw_fac * self._cov.sqrt()).clamp(self._lowest_std, INFTY)

        return _jitter(self._means[indices], std)

    def get_ess(self, w):
        return get_ess(w, normalized=True)


class ShrinkingKernel(KernelDensityEstimate):
    def fit(self, x, w):
        ess = self.get_ess(w)
        self._bw_fac = (1.59 * ess ** (-1 / 3)).clamp(EPS, 1 - EPS)

        mean = (w.unsqueeze(-1) * x).sum(0)
        self._cov = robust_var(x, w, mean)

        beta = sqrt(1.0 - self._bw_fac ** 2)
        self._means = mean + beta * (x - mean)

        return self


class NonShrinkingKernel(ShrinkingKernel):
    def fit(self, x, w):
        ess = self.get_ess(w)
        self._bw_fac = (1.59 * ess ** (-1 / 3)).clamp(EPS, 1 - EPS)

        self._cov = robust_var(x, w)
        self._means = x

        return self


class LiuWestShrinkage(ShrinkingKernel):
    def __init__(self, delta=0.98):
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
        ess = self.get_ess(w)
        self._bw_fac = self._fac(x.shape[-1], ess)

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
