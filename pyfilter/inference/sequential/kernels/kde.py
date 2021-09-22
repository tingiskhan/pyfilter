import torch
from math import sqrt
from typing import Union
from abc import ABC
from ....utils import get_ess
from ....constants import EPS, INFTY


def _jitter(values: torch.Tensor, scale: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Jitters values by the transform:
        .. math::
            \hat{\\theta} = \\theta + \epsilon, \: \epsilon \sim \mathcal{N}(0, \sigma).

    Args:
        values: The values to jitter, i.e. :math:`\\theta`.
        scale: The scale of the normal distribution used for jittering.

    Returns:
        Jittered values.
    """

    return values + scale * torch.empty_like(values).normal_()


def silverman(n: int, ess: float) -> float:
    """
    Returns Silverman's factor for KDE approximation.

    Args:
        n: The dimension of the space to construct the KDE for.
        ess: The ESS of the samples.
    """

    return (ess * (n + 2) / 4) ** (-1 / (n + 4))


def scott(n: int, ess: float):
    """
    Returns Scott's factor for KDE approximation.

    Args:
        n: See ``silverman``.
        ess: See ``silverman``.
    """

    return 1.059 * ess ** (-1 / (n + 4))


def robust_var(x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor = None) -> torch.Tensor:
    """
    Calculates the scale robustly by defining variance as:
        .. math::
            V(\\theta) = \{ \min(IQR(\\theta), \sigma(\\theta)) \}^2.

    Args:
        x: The samples to calculate the variance for.
        w: The normalized weights associated with ``x``.
        mean: Optional parameter. If you've already caluclated the mean outside of the function, you may pass that value
            to avoid wasting computational resources.
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


class KernelDensityEstimate(ABC):
    """
    Abstract base class for kernel density estimates.
    """

    def __init__(self, std_threshold: float = EPS):
        """
        Initializes ``KernelDensityEstimate`` class.

        Args:
            std_threshold: Optional parameter. The minimum allowed standard deviation to avoid issues relating to
                numerical precision whenever the KDE is "badly conditioned".
        """

        self._cov = None
        self._bw_fac = None
        self._values = None

        self._lowest_std = std_threshold

    def fit(self, x: torch.Tensor, w: torch.Tensor, indices: torch.Tensor):
        """
        Method to be overridden by derived subclasses. Specifies how to construct the KDE.

        Args:
            x: The samples to use for constructing the KDE.
            w: The normalized weights associated with ``x``.
            indices: The rows of ``x`` to choose.
        """

        raise NotImplementedError()

    def sample(self) -> torch.Tensor:
        """
        Samples from the KDE using normal distributions.
        """

        std = (self._bw_fac * self._cov.sqrt()).clamp(self._lowest_std, INFTY)

        return _jitter(self._values, std)

    def get_ess(self, w):
        return get_ess(w, normalized=True)


class ShrinkingKernel(KernelDensityEstimate):
    """
    Defines the shrinking kernel defined in `Learning and filtering via simulation: smoothly jittered particle filters.`
    by Thomas Flury and Neil Shepard.
    """

    def fit(self, x, w, indices):
        ess = self.get_ess(w)
        self._bw_fac = (1.59 * ess ** (-1 / 3)).clamp(EPS, 1 - EPS)

        mean = (w.unsqueeze(-1) * x).sum(0)
        self._cov = robust_var(x, w, mean)

        beta = sqrt(1.0 - self._bw_fac ** 2)
        self._values = (mean + beta * (x - mean))[indices]


class NonShrinkingKernel(ShrinkingKernel):
    """
    Defines the non-shrinking version of ``ShrinkingKernel``.
    """

    def fit(self, x, w, indices):
        ess = self.get_ess(w)
        self._bw_fac = (1.59 * ess ** (-1 / 3)).clamp(EPS, 1 - EPS)

        self._cov = robust_var(x, w)
        self._values = x[indices]


class LiuWestShrinkage(ShrinkingKernel):
    """
    Defines the Liu-West shrinkage kernel found in `Combined parameter and state estimation in simulation-based
    filtering` by Jane Liu and Mike West.
    """

    def __init__(self, a=0.98):
        """
        Initializes the ``LiuWestShrinkage`` class.

        Args:
             a: The ``a`` parameter of the shrinkage kernel, controls the amount of shrinkage applied to the mean of
                the distribution. Defined in (0, 1).
        """

        super().__init__()
        self._a = a
        self._bw_fac = sqrt(1 - a ** 2)

    def fit(self, x, w, indices):
        mean = (w.unsqueeze(-1) * x).sum(0)

        self._cov = robust_var(x, w, mean)
        self._values = (x * self._a + (1 - self._a) * mean)[indices]

        return self


class IndependentGaussian(ShrinkingKernel):
    """
    Basic Gaussian KDE.
    """

    def __init__(self, factor=silverman):
        """
        Initializes the ``IndependentGaussian`` class.

        Args:
            factor: Function for calculating the bandwidth factor of the KDE.
        """

        super().__init__()
        self._fac = factor
        self._w = None

    def fit(self, x, w, indices):
        ess = self.get_ess(w)
        self._bw_fac = self._fac(x.shape[-1], ess)

        self._cov = robust_var(x, w)
        self._w = w
        self._values = x[indices]

        return self


class ConstantKernel(ShrinkingKernel):
    """
    KDE assuming constant bandwidth, used in original ``NESS`` paper.
    """

    def __init__(self, bw: Union[float, torch.Tensor]):
        """
        Initializes the ``IndependentGaussian`` class.

        Args:
            bw: The constant bandwidth/scale to use.
        """

        super().__init__()
        self._bw_fac = bw
        self._w = None

    def fit(self, x, w, indices):
        self._w = w
        self._values = x[indices]
        self._cov = torch.ones(self._values.shape[-1], device=self._values.device)

        return self
