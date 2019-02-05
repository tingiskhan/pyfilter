from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter, cudawarning
from ..utils import get_ess, normalize
from ..timeseries.parameter import Parameter
import torch
from ..resampling import systematic
from scipy.stats import chi2
from math import sqrt


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
    mean = x.mean()
    var = ((x - mean) ** 2).mean()

    # ===== Skew ===== #
    skew = ((x - mean) ** 3).mean() / var ** 1.5

    # ===== Kurtosis ===== #
    kurt = ((x - mean) ** 4).mean() / var ** 2

    jb = x.shape[0] / 6 * (skew ** 2 + 1 / 4 * (kurt - 3) ** 2)

    if chi2(2).ppf(1 - alpha) < jb:
        return False

    return True


def jitter(values, scale):
    """
    Jitters the parameters.
    :param values: The values
    :type values: torch.Tensor
    :param scale: The scaling to use for the variance of the proposal
    :type scale: float
    :return: Proposed values
    :rtype: torch.Tensor
    """

    return values + scale * torch.empty_like(values).normal_()


def continuous_jitter(parameter, w, p, ess, shrink=True):
    """
    Jitters the parameters using the optimal shrinkage of ...
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :param ess: The previous ESS
    :type ess: float
    :param shrink: Whether to shrink as well as adjusting variance
    :type shrink: bool
    :return: Proposed values
    :rtype: torch.Tensor
    """
    values = parameter.t_values

    if not shrink:
        return jitter(values, 1 / sqrt(ess ** ((p + 2) / p)))

    mean, bw = shrink_(values, w, p, ess)

    return jitter(mean, bw)


def disc_jitter(parameter, i, w, p, ess, shrink):
    """
    Jitters the parameters using discrete propagation.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param i: The indices to jitter
    :type i: torch.Tensor
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :param ess: The previous ESS
    :type ess: float
    :param shrink: Whether to shrink as well as adjusting variance
    :type shrink: bool
    :return: Proposed values
    :rtype: torch.Tensor
    """
    # TODO: This may be improved
    if i.sum() == 0:
        return parameter.t_values

    return (1 - i) * parameter.t_values + i * continuous_jitter(parameter, w, p, ess, shrink=shrink)


def shrink_(values, w, p, ess):
    """
    Shrinks the parameters towards their mean.
    :param values: The values
    :type values: torch.Tensor
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :param ess: The previous ESS
    :type ess: float
    :return: The mean of the shrunk distribution and bandwidth
    :rtype: torch.Tensor, torch.Tensor
    """
    # ===== Calculate mean ===== #
    if values.dim() > w.dim():
        w = w.unsqueeze_(-1)

    mean = (w * values).sum(0)

    # ===== Calculate STD ===== #
    # TODO: Convert function to torch
    if not normal_test(values):
        sort, _ = values.sort(0)
        std = (sort[int(0.75 * values.shape[0])] - sort[int(0.25 * values.shape[0])]) / 1.349

        var = std ** 2
    else:
        var = (w * (values - mean) ** 2).sum(0)
        std = var.sqrt()

    # ===== Calculate bandwidth ===== #
    bw = 1.59 * std * ess ** (-p / (p + 2))

    # ===== Calculate shrinkage and shrink ===== #
    beta = ((var - bw ** 2) / var).sqrt()

    return mean + beta * (values - mean), bw


class NESS(SequentialAlgorithm):
    def __init__(self, filter_, particles, threshold=0.9, continuous=True, resampling=systematic, shrink=True, p=1.):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters
        :type threshold: float
        :param continuous: Whether to use continuous or discrete jittering
        :type continuous: bool
        :param p: For controlling the variance of the jittering kernel. The greater the value, the higher the variance
        for `shrink=False`. When `shrink=True`, `p` controls the amount of shrinkage applied. The smaller the value the
        more shrinkage is applied. Note that `p=1` is recommended when using `continuous=False`.
        """

        cudawarning(resampling)

        super().__init__(filter_)

        self._filter.set_nparallel(particles)

        # ===== Weights ===== #
        self._w_rec = torch.zeros(particles)

        # ===== Algorithm specific ===== #
        self._th = threshold
        self._resampler = resampling
        self._p = p
        self._shrink = shrink

        # ===== ESS related ===== #
        self._ess = particles
        self._logged_ess = tuple()

        if isinstance(filter_, ParticleFilter):
            self._shape = particles, 1
        else:
            self._shape = particles

        # ====== Select proposal kernel ===== #
        # TODO: Need to figure out why kernels aren't working too good...
        if continuous:
            self.kernel = continuous_jitter
        else:
            self.kernel = disc_jitter

    def initialize(self):
        """
        Overwrites the initialization.
        :return: Self
        :rtype: NESS
        """

        for th in self._filter.ssm.flat_theta_dists:
            th.initialize(self._shape)

        self._filter.initialize()

        return self

    @property
    def logged_ess(self):
        """
        Returns the logged ESS.
        :rtype: torch.Tensor
        """

        return torch.tensor(self._logged_ess)

    def _update(self, y):
        # ===== Resample ===== #
        self._ess = get_ess(self._w_rec)

        if self._ess < self._th * self._w_rec.shape[0]:
            indices = self._resampler(self._w_rec)
            self.filter = self.filter.resample(indices, entire_history=False)

            self._w_rec *= 0.

        # ===== Log ESS ===== #
        self._logged_ess += (self._ess,)

        # ===== Jitter ===== #
        glob_ess = self._ess
        if isinstance(self.filter, ParticleFilter):
            # TODO: Not too sure about this, but think it's correct
            glob_ess = get_ess((self._w_rec[:, None] + self.filter._w_old).view(-1))

        if self.kernel is disc_jitter:
            i = torch.empty(self._shape).bernoulli_(1 / self._ess ** (self._p / 2))
            f = lambda x: self.kernel(x, i, normalize(self._w_rec), self._p, glob_ess, self._shrink)
        else:
            f = lambda x: self.kernel(x, normalize(self._w_rec), self._p, glob_ess, self._shrink)

        self.filter.ssm.p_apply(f, transformed=True)

        # ===== Propagate filter ===== #
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        return self
