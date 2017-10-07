from .base import BaseFilter
from math import sqrt
from ..utils.utils import choose, loglikelihood
import scipy.stats as stats
import numpy as np
from ..utils.normalization import normalize


def _shrink(parameter, shrink):
    """
    Helper class for shrink parameters
    :param parameter: 
    :param shrink: 
    :return: 
    """

    return shrink * parameter + (1 - shrink) * parameter.mean()


def _propose(parameters, indices, h, particles, weights):
    """
    Helper class for proposing a parameter
    :param bounds:
    :param params:
    :param indices:
    :param weights:
    :return:
    """

    params = parameters[0]
    bounds = parameters[1].bounds()

    normalized = normalize(weights)

    means = _shrink(params[indices], sqrt(1 - h ** 2))
    std = h * np.sqrt(np.average((params - params.mean()) ** 2, weights=normalized, axis=0))

    a = (bounds[0] - means) / std
    b = (bounds[1] - means) / std

    return stats.truncnorm.rvs(a, b, means, std, size=particles)


class RAPF(BaseFilter):

    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def filter(self, y):
        self._copy = self._model.copy()
        self._copy.p_apply(lambda u: _shrink(u[0], self.a))

        t_x = self._copy.propagate_apf(self._old_x)
        t_weights = self._copy.weight(y, t_x)

        resampled_indices = self._resamp(t_weights + self._old_w)

        self._model.p_apply(lambda u: _propose(u, resampled_indices, self.h, self._p_particles, t_weights))

        resampled_x = choose(self._old_x, resampled_indices)
        x = self._proposal.draw(y, resampled_x)
        self._old_w = self._proposal.weight(y, x, resampled_x)

        self._old_x = x

        self.s_l.append(loglikelihood(self._old_w))
        self.s_mx.append(x.mean(axis=-1))

        if self.saveall:
            self.s_w.append(self._old_w - choose(t_weights, resampled_indices))
            self.s_x.append(x)

        return self