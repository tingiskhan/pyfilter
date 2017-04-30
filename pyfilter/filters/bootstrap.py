from .base import BaseFilter
import pyfilter.helpers.resampling as resamp
import pyfilter.helpers.helpers as helps


class Bootstrap(BaseFilter):
    def filter(self, y):
        t_x = self._model.propagate(self._old_x, self.parameters)
        weights = self._model.weight(y, t_x, self.parameters)

        resampled_indices = resamp.systematic(weights)

        self._old_x = helps.choose(t_x, resampled_indices)
        self._old_y = y