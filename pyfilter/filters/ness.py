from .base import BaseFilter, ParticleFilter, KalmanFilter
from .sisr import SISR
from ..utils.normalization import normalize
from ..utils.utils import get_ess
from ..distributions.continuous import Distribution
import math
import numpy as np
from scipy.stats import truncnorm, bernoulli
from .rapf import _propose


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
    std = 1 / math.sqrt(values.size ** ((p + 2) / p))

    return values + np.random.normal(scale=std, size=values.shape)


def shrink_jitter(params, p, w, a, **kwargs):
    """
    Jitters the parameters using the same shrinkage kernel as in the RAPF.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: Distribution
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :param w: The weights to use
    :type w: np.ndarray
    :param a: The `a` to use for shrinking
    :type a: float
    :return: Proposed values
    :rtype: np.ndarray
    """

    h = np.sqrt(1 - a ** 2)

    return _propose(params, range(params.values.size), h, w)


def disc_jitter(params, p, *args, **kwargs):
    """
    Jitters the parameters using discrete propagation.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: Distribution
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :return: Proposed values
    :rtype: np.ndarray
    """
    # TODO: Think about a better way to do this
    prob = 1 / params.values.shape[0] ** (p / 2)
    i = bernoulli(prob).rvs(size=params.values.shape)

    std = params.values.std()

    low, high = params.bounds()
    a, b = (low - params.values) / std, (high - params.values) / std

    return (1 - i) * params.values + i * truncnorm(a=a, b=b, loc=params.values, scale=std).rvs(size=params.values.shape)


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


class NESS(BaseFilter):
    def __init__(self, model, particles, filt=SISR, threshold=0.95, shrinkage=0.99, p=4, **filtkwargs):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param model: See BaseFilter
        :param particles: See BaseFilter
        :type particles: (int, int)|(int,)
        :param args: See BaseFilter
        :param filt: See BaseFilter
        :param threshold: The threshold for when to resample the parameters.
        :param p: A parameter controlling the variance of the jittering kernel. The greater the value, the higher the
                  variance.
        :param filtkwargs: See BaseFilter
        """
        # TODO: Perhaps change behaviour s.t. we pass an instantiated filter?
        super().__init__(model, particles)

        self._filter = filt(self._model, particles=particles, **filtkwargs).initialize()

        msg = """
        `particles` must be `tuple` or `list` of length {:d} where the first element is the number of particles 
        targeting the parameters {:s}
        """

        if isinstance(self._filter, ParticleFilter):
            if not isinstance(particles, (tuple, list)) or len(particles) != 2:
                raise ValueError(msg.format(2, 'and the second element the number of particles targeting the states'))
        elif isinstance(self._filter, KalmanFilter):
            if not isinstance(particles, (tuple, list)) or len(particles) != 1:
                raise ValueError(msg.format(1, ''))

        self._recw = 0  # type: np.ndarray
        self._th = threshold
        self._p = p

        self.a = (3 * shrinkage - 1) / 2 / shrinkage if shrinkage is not None else None
        self.h = np.sqrt(1 - self.a ** 2) if shrinkage is not None else None

        self.kernel = shrink_jitter if shrinkage is not None else cont_jitter

    def initialize(self):
        """
        Overwrites the initialization.
        :return: 
        """

        return self

    def filter(self, y):
        if isinstance(self._recw, np.ndarray):
            prev_weight = normalize(self._recw)
        else:
            prev_weight = np.ones(self._p_particles)

        # ===== JITTER ===== #

        self._model.p_apply(lambda x: self.kernel(x, self._p, prev_weight, a=self.a), transformed=True)

        # ===== PROPAGATE FILTER ===== #

        self._filter.filter(y)

        # ===== RESAMPLE PARTICLES ===== #

        self._recw += self._filter.s_l[-1]

        ess = get_ess(self._recw)

        if ess < self._th * self._filter._particles[0]:
            indices = self._resamp(self._recw)
            self._filter = self._filter.resample(indices)

            self._recw = 0  # type: np.ndarray

        return self

    def predict(self, steps, **kwargs):
        xp, yp = self._filter.predict(steps, **kwargs)

        xout = list()
        yout = list()

        for xt, yt in zip(xp, yp):
            xout.append([flattener(x) for x in xt])
            yout.append(flattener(yt))

        return xout, yout

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