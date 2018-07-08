from .base import BaseFilter, ParticleFilter, KalmanFilter
from .sisr import SISR
from ..utils.normalization import normalize
from ..utils.utils import get_ess
from ..distributions.continuous import Distribution
import math
import numpy as np


def jitter(params, p, ess):
    """
    Jitters the parameters.
    :param params: The parameters of the model, inputs as (values, prior)
    :type params: (np.ndarray, Distribution)
    :param p: The scaling to use for the variance of the proposal
    :type p: int|float
    :param ess: The effective sample size. Used for increasing/decreasing the variance of the jittering kernel
    :type ess: float
    :return: Proposed values
    :rtype: np.ndarray
    """

    transformed = params[1].transform(params[0])
    std = transformed.shape[0] / ess / math.sqrt(params[0].size ** ((p + 2) / p))

    return params[1].inverse_transform(np.random.normal(transformed, std, size=params[0].shape))


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
    def __init__(self, model, particles, filt=SISR, threshold=0.9, p=4, **filtkwargs):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param model: See BaseFilter
        :param particles: See BaseFilter
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

        if isinstance(self._filter, ParticleFilter):
            if not isinstance(particles, (tuple, list)) or len(particles) != 2:
                raise ValueError('`particles` must be `tuple` or `list` of length 2!')
        elif isinstance(self._filter, KalmanFilter):
            if not isinstance(particles, (tuple, list)) or len(particles) != 1:
                raise ValueError('`particles` must be `tuple` or `list` of length 1!')

        self._recw = 0  # type: np.ndarray
        self._th = threshold
        self._p = p

    def initialize(self):
        """
        Overwrites the initialization.
        :return: 
        """

        return self

    def filter(self, y):
        if isinstance(self._recw, np.ndarray):
            prev_ess = get_ess(self._recw)
        else:
            prev_ess = self._p_particles[0]

        # ===== JITTER ===== #

        self._model.p_apply(lambda x: jitter(x, self._p, prev_ess))

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