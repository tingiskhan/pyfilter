from .base import SequentialAlgorithm, enforce_tensor
from ..filters.base import ParticleFilter
from ..utils import get_ess, normalize
from ..timeseries.parameter import Parameter
from torch.distributions import Bernoulli
import torch
from ..resampling import systematic
import math


def cont_jitter(params, scale):
    """
    Jitters the parameters.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: Distribution
    :param scale: The scaling to use for the variance of the proposal
    :type scale: float
    :return: Proposed values
    :rtype: np.ndarray
    """
    values = params.t_values

    return values + scale * torch.empty_like(values).normal_()


def shrinkage_jitter(parameter, w, p):
    """
    Jitters the parameters using the optimal shrinkage of ...
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :return: Proposed values
    :rtype: torch.Tensor
    """
    values = parameter.t_values

    ess = get_ess(w, normalized=True)

    if values.dim() > w.dim():
        w = w.unsqueeze_(-1)

    mean = (w * values).sum(0)

    sort, _ = values.sort(0)
    std = (sort[int(0.75 * values.shape[0])] - sort[int(0.25 * values.shape[0])]) / 1.349

    var = std ** 2

    bw = 1.59 * std * ess ** (-p / (p + 2))

    beta = ((var - bw ** 2) / var).sqrt()

    return mean + beta * (values - mean) + bw * torch.empty(values.shape).normal_()


def disc_jitter(parameter, i, w):
    """
    Jitters the parameters using discrete propagation.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param i: The indices to jitter
    :type i: torch.Tensor
    :param w: The normalized weights
    :type w: torch.Tensor
    :return: Proposed values
    :rtype: torch.Tensor
    """
    # TODO: This may be improved
    if i.sum() == 0:
        return parameter.t_values

    return (1 - i) * parameter.t_values + i * shrinkage_jitter(parameter, w, 1)


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
        more shrinkage is applied. Note that `p=1` is recommendended when using `continuous=False`.
        """

        super().__init__(filter_)

        self._filter.set_nparallel(particles)

        self._w_rec = torch.zeros(particles)
        self._th = threshold
        self._resampler = resampling

        if isinstance(filter_, ParticleFilter):
            self._shape = particles, 1
        else:
            self._shape = particles

        # ====== Select proposal kernel ===== #
        # TODO: Need to figure out why kernels aren't working too good...
        if continuous:
            if shrink:
                self.kernel = lambda u, w: shrinkage_jitter(u, w, p=p)
            else:
                scale = 1 / math.sqrt(particles ** ((p + 2) / p))
                self.kernel = lambda u, w: cont_jitter(u, scale)
        else:
            bernoulli = Bernoulli(1 / self._w_rec.shape[0] ** (p / 2))

            if isinstance(self._shape, tuple):
                self.sampler = lambda: bernoulli.sample(self._shape)
            else:
                self.sampler = lambda: bernoulli.sample((self._shape, 1))[..., 0]

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

    @enforce_tensor
    def update(self, y):
        # ===== Propagate filter ===== #
        self.filter.filter(y)
        self._w_rec += self._filter.s_ll[-1]

        # ===== Jitter ===== #
        if self.kernel is disc_jitter:
            i = self.sampler()
            self._filter.ssm.p_apply(lambda x: self.kernel(x, i, normalize(self._w_rec)), transformed=True)
        else:
            self._filter.ssm.p_apply(lambda x: self.kernel(x, normalize(self._w_rec)), transformed=True)

        # ===== Resample ===== #
        ess = get_ess(self._w_rec)

        if ess < self._th * self._w_rec.shape[0]:
            indices = self._resampler(self._w_rec)
            self._filter = self.filter.resample(indices, entire_history=False)

            self._w_rec *= 0.

        return self
