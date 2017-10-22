from .base import BaseFilter
import pyfilter.utils.utils as helps
from ..utils.normalization import normalize
import numpy as np


class SISR(BaseFilter):
    def filter(self, y):
        t_x = self._proposal.draw(y, self._old_x, size=self._particles)
        weights = self._proposal.weight(y, t_x, self._old_x)

        resampled_indices = self._resamp(weights)

        self._cur_x = t_x
        self._inds = resampled_indices
        self._anc_x = self._old_x.copy()
        self._old_x = helps.choose(t_x, resampled_indices)
        self._old_w = weights

        self.s_l.append(helps.loglikelihood(weights))
        self.s_mx.append(np.sum(t_x * normalize(weights), axis=-1))

        if self.saveall:
            self.s_x.append(t_x)
            self.s_w.append(weights)

        return self