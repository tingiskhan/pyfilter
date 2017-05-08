from .base import BaseFilter
from ..distributions.continuous import Distribution
import numpy as np
from ..helpers.normalization import normalize
from math import sqrt
from ..helpers.resampling import systematic
from ..helpers.helpers import choose


class RAPF(BaseFilter):

    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def _initialize_parameters(self):
        self._topropagate = list()

        for ts in self._model.hidden:
            out = tuple()
            out2 = tuple()
            for i, p in enumerate(ts.theta):
                if isinstance(p, Distribution):
                    out += (p.rvs(size=self._particles),)
                    out2 += (i,)
                else:
                    out += (ts.theta[i],)

            self._topropagate.append(out2)
            ts.theta = out

        return self

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)

        return self

    def _propagate_parameters(self, indices, weights):
        normalized = normalize(weights)

        for i, ts in enumerate(self._model.hidden):
            out = tuple()
            for j, p in enumerate(ts.theta):
                if j in self._topropagate[i]:
                    average = np.average(p, weights=normalized)

                    mean = self.a * p[indices] + (1 - self.a) * average
                    variance = np.average((p - average) ** 2, weights=normalized)

                    t_p = np.random.normal(mean, self.h * np.sqrt(variance), size=p.shape)
                    # TODO: Add step where parameter is truncated to its actual space
                else:
                    t_p = p

                out += (t_p,)

            ts.theta = out

        return self

    def filter(self, y):
        t_x = self._model.propagate_apf(self._old_x)
        t_w = self._model.weight(y, t_x)

        try:
            resampled_indices = systematic(t_w + self.s_w[-1])
        except IndexError:
            resampled_indices = systematic(t_w)

        self._propagate_parameters(resampled_indices, t_w)

        self._old_x = self._model.propagate(choose(self._old_x, resampled_indices))
        weights = self._model.weight(y, self._old_x)

        self._old_y = y

        self.s_x.append(t_x)
        self.s_w.append(weights - choose(t_w, resampled_indices))
        self.s_l.append(weights.mean(axis=-1))

        return self
