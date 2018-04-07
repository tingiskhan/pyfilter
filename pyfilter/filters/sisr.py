from .base import BaseFilter
from ..utils.utils import loglikelihood, choose
from ..utils.normalization import normalize
import numpy as np


class SISR(BaseFilter):
    """
    Implements the SISR filter by Gordon et al.
    """
    def filter(self, y):
        t_x = self._proposal.draw(y, self._old_x, size=self._particles)
        weights = self._proposal.weight(y, t_x, self._old_x)

        resampled_indices = self._resamp(weights)

        self._proposal = self._proposal.resample(resampled_indices)
        self._cur_x = t_x
        self._inds = resampled_indices
        self._anc_x = self._old_x.copy()
        self._old_x = choose(t_x, resampled_indices)
        self._old_w = weights

        self.s_l.append(loglikelihood(weights))

        if self.saveall:
            self.s_x.append(t_x)
            self.s_w.append(weights)

        return self._save_mean_and_noise(y, t_x, normalize(weights))