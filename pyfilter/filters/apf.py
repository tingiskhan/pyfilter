from .base import BaseFilter
import pyfilter.helpers.resampling as resamp
import pyfilter.helpers.helpers as helps


class APF(BaseFilter):
    def filter(self, y):
        t_x = self._model.propagate_apf(self._old_x)
        t_weights = self._model.weight(y, t_x)

        try:
            resampled_indices = resamp.systematic(t_weights + self.s_w[-1])
        except IndexError:
            resampled_indices = resamp.systematic(t_weights)

        resampled_x = helps.choose(self._old_x, resampled_indices)
        t_x = self._model.propagate(resampled_x)
        weights = self._model.weight(y, t_x)

        self._old_y = y
        self._old_x = t_x

        self.s_x.append(t_x)
        self.s_w.append(weights - helps.choose(t_weights, resampled_indices))

        return self