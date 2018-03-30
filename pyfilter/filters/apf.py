from .base import BaseFilter
from ..utils.utils import loglikelihood, choose
from ..utils.normalization import normalize
import numpy as np


class APF(BaseFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """
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
        resampled_x = choose(self._old_x, resampled_indices)

        t_x = self._proposal.draw(y, resampled_x)
        weights = self._proposal.weight(y, t_x, resampled_x)

        self._cur_x = t_x
        self._inds = resampled_indices
        self._anc_x = self._old_x.copy()
        self._old_x = t_x
        self._old_w = weights - choose(t_weights, resampled_indices)

        self.s_mx.append(np.sum(t_x * normalize(self._old_w), axis=-1))

        # ===== Calculate log likelihood ===== #

        with np.errstate(divide='ignore'):
            self.s_l.append(loglikelihood(self._old_w) + np.log((normalized * np.exp(t_weights)).sum(axis=-1)))

        if self.saveall:
            self.s_x.append(t_x)
            self.s_w.append(self._old_w)

        return self