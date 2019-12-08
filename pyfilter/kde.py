import torch
from .resampling import residual
from .utils import get_ess


def _jitter(values, scale):
    """
    Jitters the parameters.
    :param values: The values
    :type values: torch.Tensor
    :param scale: The scaling to use for the variance of the proposal
    :type scale: float|torch.Tensor
    :return: Proposed values
    :rtype: torch.Tensor
    """

    return values + scale * torch.empty_like(values).normal_()


def silverman(n, ess):
    """
    Returns Silverman's factor.
    :param n: The dimension
    :type n: int
    :param ess: The ess
    :type ess: float
    :return: Bandwidth factor
    :rtype: float
    """

    return (ess * (n + 2) / 4) ** (-1 / (n + 4))


def scott(n, ess):
    """
    Returns Silverman's factor.
    :param n: The dimension
    :type n: int
    :param ess: The ess
    :type ess: float
    :return: Bandwidth factor
    :rtype: float
    """

    return 1.059 * ess ** (-1 / (n + 4))


def robust_var(x, w, mean=None):
    """
    Calculates the scale robustly
    :param x: The values
    :type x: torch.Tensor
    :param w: The weights
    :type w: torch.Tensor
    :param mean: The mean
    :type mean: torch.Tensor
    :rtype: torch.Tensor
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
    def __init__(self, resampling=residual):
        """
        Implements the base class for KDEs.
        :param resampling: The resampler function
        :type resampling: callable
        """

        self._resampling = resampling
        self._cov = None
        self._bw_fac = None
        self._means = None

    def fit(self, x, w):
        """
        Fits the KDE.
        :param x: The samples
        :type x: torch.Tensor
        :param w: The weights
        :type w: torch.Tensor
        :return: Self
        :rtype: KernelDensityEstimate
        """

        raise NotImplementedError()

    def sample(self, inds=None):
        """
        Samples from the KDE.
        :param inds: Whether to manually specify the samples chosen
        :type inds: torch.Tensor
        :return: New samples
        :rtype: torch.Tensor
        """

        raise NotImplementedError()


class ShrinkingKernel(KernelDensityEstimate):
    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = get_ess(w)
        self._bw_fac = 1.59 * ess ** (-1 / 3)

        # ===== Calculate variance ===== #
        mean = (w.unsqueeze(-1) * x).sum(0)
        self._cov = robust_var(x, w, mean)

        # ===== Calculate shrinkage and shrink ===== #
        beta = (1. - self._bw_fac ** 2).sqrt()
        self._means = mean + beta * (x - mean)

        return self

    def sample(self, inds=None):
        inds = inds if inds is not None else torch.ones_like(self._means)
        return _jitter(self._means[inds], self._bw_fac * self._cov.sqrt())


class NonShrinkingKernel(ShrinkingKernel):
    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = get_ess(w)
        self._bw_fac = 1.59 * ess ** (-1 / 3)

        # ===== Calculate variance ===== #
        self._cov = robust_var(x, w)

        # ===== Calculate shrinkage and shrink ===== #
        self._means = x

        return self


class IndependentGaussian(KernelDensityEstimate):
    def __init__(self, factor=silverman):
        """
        Implements a Gaussian KDE.
        :param factor: How to calculate the factor bandwidth
        :type factor: callable
        """
        super().__init__()
        self._fac = factor
        self._w = None

    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = get_ess(w)
        self._bw_fac = self._fac(x.shape[-1], ess)

        # ===== Calculate covariance ===== #
        self._cov = robust_var(x, w)
        self._w = w
        self._means = x

        return self

    def sample(self, inds=None):
        inds = inds if inds is not None else self._resampling(self._w, normalized=True)

        return _jitter(self._means[inds], self._bw_fac * self._cov.sqrt())


class MultivariateGaussian(IndependentGaussian):
    def __init__(self, **kwargs):
        """
        Constructs a multivariate Gaussian kernel.
        :param kwargs: Any kwargs
        """
        super().__init__(**kwargs)
        self._post_mean = None

    def fit(self, x, w):
        # ===== Calculate bandwidth ===== #
        ess = get_ess(w)
        self._bw_fac = self._fac(x.shape[-1], ess)

        # ===== Calculate statistics ===== #
        self._w = w

        self._post_mean = mean = (x * w.unsqueeze(-1)).sum(0)
        centralized = x - mean
        self._cov = torch.matmul(w * centralized.t(), centralized).cholesky()

        self._means = torch.solve((x - self._post_mean).T, self._cov).solution.T

        return self

    def sample(self, inds=None):
        inds = inds if inds is not None else self._resampling(self._w, normalized=True)
        jittered = _jitter(self._means[inds], self._bw_fac)

        return self._post_mean + jittered.matmul(self._cov.T)