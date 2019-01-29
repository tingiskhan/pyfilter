from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter
from ..utils import get_ess, normalize
from ..timeseries.parameter import Parameter
import torch
from ..resampling import systematic
from scipy.stats import normaltest
from math import sqrt


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

    mean, beta, bw = shrink_(values, w, p, ess)

    return jitter(mean + beta * (values - mean), bw)


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
    :return: The mean, shrinkage factor and bandwidth
    :rtype: torch.Tensor, torch.Tensor
    """
    # ===== Calculate mean ===== #
    if values.dim() > w.dim():
        w = w.unsqueeze_(-1)

    mean = (w * values).sum(0)

    # ===== Calculate STD ===== #
    # TODO: Convert function to torch
    if normaltest(values)[-1] < 0.05:
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

    return mean, beta, bw


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

        super().__init__(filter_)

        self._filter.set_nparallel(particles)

        self._w_rec = torch.zeros(particles)
        self._th = threshold

        self._resampler = resampling
        self._ess = particles
        self._p = p
        self._shrink = shrink

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

    def _update(self, y):
        # ===== Jitter ===== #
        if self.kernel is disc_jitter:
            i = torch.empty(self._shape).bernoulli_(1 / self._ess ** (self._p / 2))
            f = lambda x: self.kernel(x, i, normalize(self._w_rec), self._ess, self._shrink)
        else:
            f = lambda x: self.kernel(x, normalize(self._w_rec), self._p, self._ess, self._shrink)

        self._filter.ssm.p_apply(f, transformed=True)

        # ===== Propagate filter ===== #
        self.filter.filter(y)
        self._w_rec += self._filter.s_ll[-1]

        # ===== Resample ===== #
        self._ess = get_ess(self._w_rec)

        if self._ess < self._th * self._w_rec.shape[0]:
            indices = self._resampler(self._w_rec)
            self._filter = self.filter.resample(indices, entire_history=False)

            self._w_rec *= 0.

        return self
