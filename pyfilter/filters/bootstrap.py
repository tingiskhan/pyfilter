from .base import BaseFilter
import pyfilter.helpers.resampling as resamp
import pyfilter.helpers.helpers as helps


class Bootstrap(BaseFilter):
    def filter(self, y):
        t_x = self._model.propagate(self._old_x)
        weights = self._model.weight(y, t_x)

        resampled_indices = resamp.systematic(weights)

        self._old_x = helps.choose(t_x, resampled_indices)
        self._old_y = y

        self.s_x.append(t_x)
        self.s_w.append(weights)
        self.s_l.append(weights.mean(axis=-1))

        return self