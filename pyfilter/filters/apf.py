from .base import ParticleFilter
from ..utils.utils import loglikelihood, choose
from ..utils.normalization import normalize
import numpy as np


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """
    def _filter(self, y):
        # ===== Perform auxiliary sampling ===== #
        t_x = self._model.propagate_apf(self._x_cur)
        t_weights = self._model.weight(y, t_x)

        resamp_w = t_weights + self._w_old
        normalized = normalize(self._w_old)

        # ===== Resample and propagate ===== #
        resampled_indices = self._resamp(resamp_w)
        resampled_x = choose(self._x_cur, resampled_indices)

        t_x = self._proposal.draw(y, resampled_x)
        weights = self._proposal.weight(y, t_x, resampled_x)

        self._cur_x = t_x
        self._w_old = weights - choose(t_weights, resampled_indices)

        # ===== Calculate log likelihood ===== #
        ll = loglikelihood(self._w_old) + np.log((normalized * np.exp(t_weights)).sum(-1))

        return (normalize(self._w_old) * t_x).sum(-1), ll