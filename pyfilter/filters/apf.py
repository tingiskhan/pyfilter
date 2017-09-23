from .base import BaseFilter
import pyfilter.utils.utils as helps
from ..utils.normalization import normalize
import numpy as np


class APF(BaseFilter):
    def filter(self, y):

        # ===== Perform "auxiliary sampling ===== #

        t_x = self._model.propagate_apf(self._old_x)
        t_weights = self._model.weight(y, t_x)

        if not isinstance(self._old_w, int):
            resamp_w = t_weights + self._old_w
            normalized = normalize(self._old_w)
        else:
            resamp_w = t_weights
            normalized = 1 / t_weights.shape[-1]

        # ===== Resample and propagate ===== #

        resampled_indices = self._resamp(resamp_w)
        resampled_x = helps.choose(self._old_x, resampled_indices)

        t_x = self._proposal.draw(y, resampled_x)
        weights = self._proposal.weight(y, t_x, resampled_x)

        self._old_y = y
        self._old_x = t_x
        self._old_w = weights - helps.choose(t_weights, resampled_indices)

        n_normalized = normalize(self._old_w)

        self.s_mx.append([np.sum(x * n_normalized, axis=-1) for x in t_x])

        # ===== Calculate log likelihood ===== #

        with np.errstate(divide='ignore'):
            self.s_l.append(helps.loglikelihood(self._old_w) + np.log((normalized * np.exp(t_weights)).sum(axis=-1)))

        if self.saveall:
            self.s_x.append(t_x)
            self.s_w.append(self._old_w)

        return self