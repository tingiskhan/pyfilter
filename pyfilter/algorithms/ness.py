from .base import SequentialAlgorithm, enforce_tensor
from ..filters.base import ParticleFilter
from ..utils import get_ess
from ..timeseries.parameter import Parameter
import math
import numpy as np
from torch.distributions import Bernoulli
import torch
from ..resampling import systematic


def cont_jitter(parameter, scale, *args):
    """
    Jitters the parameters.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param scale: The scale to use for propagating the parameters
    :type scale: float
    :return: Proposed values
    :rtype: torch.Tensor
    """
    # TODO: Can we improve the jittering kernel?
    values = parameter.t_values

    return values + scale * torch.empty(values.shape).normal_()


def disc_jitter(parameter, i, *args):
    """
    Jitters the parameters using discrete propagation.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param h: The `h` to use for shrinking
    :type h: float
    :param i: The indices to jitter
    :type i: torch.Tensor
    :return: Proposed values
    :rtype: torch.Tensor
    """
    # TODO: Check if this even makes sense
    transformed = parameter.t_values
    std = 1.06 * transformed.std() * transformed.shape[0] ** (-1 / 5)

    indices = torch.multinomial(torch.ones(transformed.shape[0]), num_samples=int(i.sum()))
    propagated = transformed[indices] + std * torch.empty_like(transformed[indices]).normal_()

    # ===== Define out ===== #
    # TODO: Fix this
    transformed[i.byte()] = propagated[..., 0]

    return transformed


def flattener(a):
    """
    Flattens array a.
    :param a: An array
    :type a: np.ndarray
    :return: Flattened array
    :rtype: np.ndarray
    """

    if a.ndim < 3:
        return a.flatten()

    return a.reshape(a.shape[0], a.shape[1] * a.shape[2])


class NESS(SequentialAlgorithm):
    def __init__(self, filter_, particles, threshold=0.9, continuous=False, resampler=systematic, p=4):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters
        :type threshold: float
        :param continuous: Whether to use continuous or discrete jittering
        :type continuous: bool
        :param p: For controlling the variance of the jittering kernel. The greater the value, the higher the variance.
        """

        super().__init__(filter_)

        self._filter.set_nparallel(particles)

        self._w_rec = torch.zeros(particles)
        self._th = threshold
        self._resampler = resampler

        if isinstance(filter_, ParticleFilter):
            self._shape = particles, 1
        else:
            self._shape = particles

        if continuous:
            scale = 1 / math.sqrt(particles ** ((p + 2) / p))
            self.kernel = lambda u, w: cont_jitter(u, scale, w)
        else:
            bernoulli = Bernoulli(1 / self._w_rec.shape[0] ** (p / 2))
            self.kernel = lambda u: disc_jitter(u, i=bernoulli.sample(self._shape))

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
        # ===== Jitter ===== #
        self._filter.ssm.p_apply(lambda x: self.kernel(x), transformed=True)

        # ===== Propagate filter ===== #
        self.filter.filter(y)

        # ===== Resample ===== #
        self._w_rec += self._filter.s_ll[-1]

        ess = get_ess(self._w_rec)

        if ess < self._th * self._w_rec.shape[0]:
            indices = self._resampler(self._w_rec)
            self._filter = self.filter.resample(indices, entire_history=False)

            self._w_rec *= 0.

        return self
