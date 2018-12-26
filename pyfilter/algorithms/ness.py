from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter
from ..normalization import normalize
from ..utils import get_ess
from ..timeseries.parameter import Parameter
import math
import numpy as np
from torch.distributions import Bernoulli, Normal
import torch
from math import sqrt


def cont_jitter(parameter, p, *args):
    """
    Jitters the parameters.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :return: Proposed values
    :rtype: torch.Tensor
    """
    # TODO: Can we improve the jittering kernel?
    values = parameter.t_values
    std = 1 / math.sqrt(values.shape[0] ** ((p + 2) / p))

    return values + std * torch.empty(values.shape).normal_()


def disc_jitter(parameter, p, w, h, i, *args):
    """
    Jitters the parameters using discrete propagation.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :param w: The weights to use
    :type w: torch.Tensor
    :param h: The `h` to use for shrinking
    :type h: float
    :param i: The indices to jitter
    :type i: torch.Tensor
    :return: Proposed values
    :rtype: torch.Tensor
    """
    normalized = normalize(w)[..., None]
    transformed = parameter.t_values

    weighted_mean = (transformed * normalized).sum(0)

    # ===== Shrink ===== #
    a = sqrt(1 - h ** 2)
    means = a * transformed + (1 - a) * weighted_mean
    std = h * torch.sqrt(((transformed - weighted_mean) ** 2).mean())

    return (1 - i) * transformed + i * Normal(means, std).sample()


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
    def __init__(self, filter_, particles, threshold=0.9, shrinkage=None, p=4):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters.
        :param p: For controlling the variance of the jittering kernel. The greater the value, the higher the variance.
        """

        super().__init__(filter_)

        self._filter.set_nparallel(particles)

        self._w_rec = torch.zeros(particles)
        self._th = threshold
        self._p = p

        if isinstance(filter_, ParticleFilter):
            self._shape = particles, 1
        else:
            self._shape = particles

        self.a = (3 * shrinkage - 1) / 2 / shrinkage if shrinkage is not None else None
        self.h = sqrt(1 - self.a ** 2) if shrinkage is not None else None

        if shrinkage is None:
            self.kernel = lambda u, w: cont_jitter(u, self._p, w)
        else:
            bernoulli = Bernoulli(1 / self._w_rec.shape[0] ** (self._p / 2))
            self.kernel = lambda u, w: disc_jitter(u, self._p, w, h=self.h, i=bernoulli.sample(self._shape))

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

    def update(self, y):
        # ===== Jitter ===== #
        self._filter.ssm.p_apply(lambda x: self.kernel(x, self._w_rec), transformed=True)

        # ===== Propagate filter ===== #
        self.filter.filter(y)

        # ===== Resample ===== #
        self._w_rec += self._filter.s_ll[-1]

        ess = get_ess(self._w_rec)

        if ess < self._th * self._w_rec.shape[0]:
            indices = self.filter._resampler(self._w_rec)
            self._filter = self.filter.resample(indices, entire_history=False)

            self._w_rec = torch.zeros_like(self._w_rec)

        return self

    def predict(self, steps):
        xp, yp = self.filter.predict(steps)

        xout = list()
        yout = list()

        for xt, yt in zip(xp, yp):
            xout.append([flattener(x) for x in xt])
            yout.append(flattener(yt))

        return np.array(xout), np.array(yout)

    def filtermeans(self):
        out = list()
        for tw, tx in zip(self.filter.s_ll, self._filter.s_mx):
            normalized = normalize(tw)
            out.append(np.sum(tx * normalized, axis=-1))

        return out