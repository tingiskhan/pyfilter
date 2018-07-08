from .base import BaseFilter
from math import sqrt
from ..utils.utils import choose, loglikelihood
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
    transformed = parameters[1].transform(parameters[0])

    normalized = normalize(weights)

    means = _shrink(transformed, sqrt(1 - h ** 2), weights)[indices]
    mean = np.average(transformed, weights=normalized)
    std = h * np.sqrt(np.average((transformed - mean) ** 2, weights=normalized))

    return parameters[1].inverse_transform(np.random.normal(means, std, size=particles))


class RAPF(BaseFilter):
    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):
        """
        Implements the Regularized Auxiliary Particle Filter for parameter inference by Liu and West.
        :param model: See BaseFilter
        :param particles: See BaseFilter
        :type particles: int
        :param args: See BaseFilter
        :param shrinkage: The shrinkage to use
        :type shrinkage: float
        :param kwargs:
        """

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def filter(self, y):
        if not isinstance(self._old_w, np.ndarray):
            self._old_w = 1 / self._particles * np.ones(self._particles)

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