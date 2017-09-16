from .base import BaseFilter
from .sisr import SISR
from ..utils.normalization import normalize
from ..utils.utils import get_ess
import scipy.stats as stats
import math
import numpy as np


def jitter(params, p):
    """
    Jitters the parameters.
    :param params:
    :param p:
    :return: 
    """
    std = params[1].std() / math.sqrt(params[0].size ** ((p + 2) / p))

    a = (params[1].bounds()[0] - params[0]) / std
    b = (params[1].bounds()[1] - params[0]) / std

    return stats.truncnorm.rvs(a, b, params[0], std, size=params[0].shape)


def flattener(a):
    """
    Flattens array a.
    :param a: 
    :return: 
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

        super().__init__(model, particles)

        self._filter = filt(self._model, particles, **filtkwargs).initialize()
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

        # ===== JITTER ===== #

        self._model.p_apply(lambda x: jitter(x, self._p))

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
        return [x.mean(axis=-1) for tx in self._filter.s_mx for x in tx]