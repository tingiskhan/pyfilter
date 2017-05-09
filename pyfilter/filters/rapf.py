from .base import BaseFilter
from ..distributions.continuous import Distribution
import numpy as np
from ..helpers.normalization import normalize
from math import sqrt
from ..helpers.resampling import systematic
from ..helpers.helpers import choose
import scipy.stats as stats


class RAPF(BaseFilter):

    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def _initialize_parameters(self):
        # TODO: Rewrite

        return self

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)

        return self

    def filter(self, y):
        # TODO: Implement a working filter
        pass