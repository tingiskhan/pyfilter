from .base import BaseFilter
from math import sqrt
from ..utils.utils import choose, loglikelihood
import scipy.stats as stats
import numpy as np
from ..utils.normalization import normalize


def _shrink(parameter, shrink, weights):
    """
    Helper class for shrink parameters
    :param parameter: 
    :param shrink:
    :param weights:
    :return: 
    """

    return shrink * parameter + (1 - shrink) * np.average(parameter, weights=normalize(weights))


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

    means = _shrink(params, sqrt(1 - h ** 2), weights)[indices]
    mean = np.average(params, weights=normalized)
    std = h * np.sqrt(np.average((params - mean) ** 2, weights=normalized))

    a = (bounds[0] - means) / std
    b = (bounds[1] - means) / std

    return stats.truncnorm.rvs(a, b, means, std, size=particles)


class RAPF(BaseFilter):

    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def filter(self, y):
        if not isinstance(self._old_w, np.ndarray):
            self._old_w = 1 / self._particles * np.ones_like(self._old_x)

        # ==== Propagate APF ===== #
        copy = self._model.copy()
        t_x = copy.propagate_apf(self._old_x)
        copy.p_apply(lambda u: _shrink(u[0], self.a, self._old_w))

        # ===== Weight and get indices ===== #
        t_weights = copy.weight(y, t_x)
        resampled_indices = self._resamp(t_weights + self._old_w)

        # ===== Propose new parameters ===== #
        self._model.p_apply(lambda u: _propose(u, resampled_indices, self.h, self._p_particles, self._old_w))

        # ===== Propagate the good states ===== #

        resampled_x = choose(self._old_x, resampled_indices)
        x = self._proposal.draw(y, resampled_x)

        # ===== Calculate the new weights ===== #

        self._old_w = self._proposal.weight(y, x, resampled_x) - choose(t_weights, resampled_indices)
        self._old_x = x

        self.s_l.append(loglikelihood(self._old_w))
        self.s_mx.append(np.sum(x * normalize(self._old_w), axis=-1))

        if self.saveall:
            self.s_w.append(self._old_w - choose(t_weights, resampled_indices))
            self.s_x.append(x)

        return self