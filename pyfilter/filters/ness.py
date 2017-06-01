from .base import BaseFilter
from .bootstrap import Bootstrap
from ..helpers.resampling import systematic
import numpy as np
import math


def jitter(params):
    """
    Jitters the parameters.
    :param params: 
    :return: 
    """
    # TODO: Fix this to use truncated
    return np.abs(params + math.sqrt(20 / params.size ** (3 / 2)) * np.random.normal(size=params.shape))


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