from .base import BaseFilter
from .bootstrap import Bootstrap
from ..helpers.resampling import systematic
import scipy.stats as stats
import math


def jitter(params):
    """
    Jitters the parameters.
    :param params: 
    :return: 
    """
    # TODO: Fix this to use truncated

    std = math.sqrt(5 / params[0].size ** (3 / 2))

    a = (params[1].bounds()[0] - params[0]) / std
    b = (params[1].bounds()[1] - params[0]) / std

    return stats.truncnorm.rvs(a, b, params[0], std, size=params[0].shape)


class NESS(BaseFilter):
    def __init__(self, model, particles, *args, filt=Bootstrap, **kwargs):
        super().__init__(model, particles, *args, **kwargs)

        self._filter = filt(model, particles).initialize()

    def initialize(self):
        """
        Overwrites the initialization.
        :return: 
        """

        return self

    def filter(self, y):

        # ===== JITTER ===== #

        self._model.p_apply(jitter)

        # ===== PROPAGATE FILTER ===== #

        self._filter.filter(y)

        # ===== RESAMPLE PARTICLES ===== #

        indices = systematic(self._filter.s_l[-1])
        self._filter = self._filter.resample(indices)

        return self

    def predict(self, steps, **kwargs):
        return self._filter.predict(steps, **kwargs)

    def filtermeans(self):
        return self._filter.filtermeans()