from .base import BaseFilter
from .bootstrap import Bootstrap
from ..helpers.resampling import systematic


def jitter(params, weights):
    """
    Jitters the parameters.
    :param params: 
    :param weights: 
    :return: 
    """


class NESS(BaseFilter):
    def __init__(self, model, particles, *args, filt=Bootstrap, **kwargs):
        super().__init__(model, particles, *args, **kwargs)

        self._filter = filt(model, particles).initialize()
        self._c_filter = self._filter.copy()

    def initialize(self):
        """
        Overwrites the initialization.
        :return: 
        """

        return self

    def filter(self, y):

        # ===== JITTER ===== #

        self._model.p_apply(lambda x: x)

        # ===== PROPAGATE FILTER ===== #

        self._filter.filter(y)

        # ===== RESAMPLE PARTICLES ===== #

        indices = systematic(self._filter.s_l[-1])
        self._filter = self._filter.resample(indices)

        return self