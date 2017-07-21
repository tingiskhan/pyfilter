from .base import BaseFilter
import numpy as np


class UPF(BaseFilter):
    lamda = 1

    _extendedmean = None
    _extendedcov = None

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)

        shape = 2 * self._model.hidden_ndim + self._model.obs_ndim

        try:
            self._extendedmean = np.empty((shape, self._particles[-2]))
            self._extendedcov = np.empty((shape, shape, self._particles[-2]))
        except (IndexError, Exception):
            self._extendedmean = np.empty((shape,))
            self._extendedcov = np.empty((shape, shape))

        return self
