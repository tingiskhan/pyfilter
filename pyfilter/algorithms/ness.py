from .base import OnlineAlgorithm
from ..filters.base import ParticleFilter
from ..utils.normalization import normalize
from ..utils.utils import get_ess
import math
import numpy as np
from scipy.stats import bernoulli
from .rapf import _propose
import torch


def cont_jitter(params, p, *args, **kwargs):
    """
    Jitters the parameters.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: Distribution
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :return: Proposed values
    :rtype: np.ndarray
    """
    # TODO: Can we improve the jittering kernel?
    values = params.t_values
    std = 1 / math.sqrt(values.shape[0] ** ((p + 2) / p))

    return values + std * torch.empty(values.shape).normal_()


def shrink_jitter(params, p, w, h, **kwargs):
    """
    Jitters the parameters using the same shrinkage kernel as in the RAPF.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: Distribution
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :param w: The weights to use
    :type w: np.ndarray
    :param h: The `a` to use for shrinking
    :type h: float
    :return: Proposed values
    :rtype: np.ndarray
    """

    return _propose(params, range(params.values.size), h, w)


def disc_jitter(params, p, w, h, i, **kwargs):
    """
    Jitters the parameters using discrete propagation.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: Distribution
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :param w: The weights to use
    :type w: np.ndarray
    :param h: The `a` to use for shrinking
    :type h: float
    :param i: The indices to jitter
    :type i: np.ndarray
    :return: Proposed values
    :rtype: np.ndarray
    """
    # TODO: Only jitter the relevant particles
    return (1 - i) * params.t_values + i * shrink_jitter(params, p, w, h)


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


class NESS(OnlineAlgorithm):
    def __init__(self, filter_, particles, threshold=0.9, shrinkage=None, p=4):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters.
        :param p: For controlling the variance of the jittering kernel. The greater the value, the higher the variance.
        """

        super().__init__(filter_)

        if isinstance(self._filter, ParticleFilter):
            self._filter.set_particles((particles, self._filter._particles))

        self._w_rec = torch.zeros(particles)
        self._th = threshold
        self._p = p

        self.a = (3 * shrinkage - 1) / 2 / shrinkage if shrinkage is not None else None
        self.h = np.sqrt(1 - self.a ** 2) if shrinkage is not None else None

        self.kernel = disc_jitter if shrinkage is not None else cont_jitter

    def initialize(self):
        """
        Overwrites the initialization.
        :return: Self
        :rtype: NESS
        """

        for th in self._filter.ssm.flat_theta_dists:
            th.initialize((self._w_rec.shape[0], 1))

        self._filter.initialize()

        return self

    def update(self, y):
        # ===== Jitter ===== #
        # TODO: Think about a better way to do this
        if self.kernel == disc_jitter:
            prob = 1 / self._w_rec.size() ** (self._p / 2)
            i = bernoulli(prob).rvs(size=self._p_particles)
        else:
            i = 0

        self._filter.ssm.p_apply(lambda x: self.kernel(x, self._p, self._w_rec, h=self.h, i=i), transformed=True)

        # ===== Propagate filter ===== #
        self._filter.filter(y)

        # ===== Resample ===== #
        self._w_rec += self._filter.s_ll[-1]

        ess = get_ess(self._w_rec)

        if ess < self._th * self._filter._particles[0]:
            indices = self._filter._resamp(self._w_rec)
            self._filter = self._filter.resample(indices, entire_history=False)

            self._w_rec = torch.zeros_like(self._w_rec)

        return self

    def predict(self, steps, **kwargs):
        xp, yp = self._filter.predict(steps, **kwargs)

        xout = list()
        yout = list()

        for xt, yt in zip(xp, yp):
            xout.append([flattener(x) for x in xt])
            yout.append(flattener(yt))

        return np.array(xout), np.array(yout)

    def filtermeans(self):
        out = list()
        for tw, tx in zip(self._filter.s_l, self._filter.s_mx):
            normalized = normalize(tw)
            out.append(np.sum(tx * normalized, axis=-1))

        return out

    def noisemeans(self):
        out = list()
        for tw, tx in zip(self._filter.s_l, self._filter.s_n):
            normalized = normalize(tw)
            out.append(np.sum(tx * normalized, axis=-1))

        return out