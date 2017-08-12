from .base import BaseFilter
import numpy as np


class UPF(BaseFilter):
    lamda = 1

    _extendedmean = None
    _extendedcov = None

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)

        shape = 2 * sum(self._model.hidden_ndim) + self._model.obs_ndim

        try:
            self._extendedmean = np.zeros((shape, self._particles[-2]))
            self._extendedcov = np.zeros((shape, shape, self._particles[-2]))
        except (IndexError, Exception):
            self._extendedmean = np.zeros((shape,))
            self._extendedcov = np.zeros((shape, shape))

        cumsummed = np.cumsum(self._model.hidden_ndim)
        for i in range(len(self._old_x)):
            sli = slice(min(0, cumsummed[i-1]), cumsummed[i])

            self._extendedmean[sli] = self._old_x[i].mean(axis=-1)
            self._extendedcov[sli, sli] = self._model.hidden[i].i_scale()

        return self
