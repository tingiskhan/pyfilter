from .base import BaseFilter
from math import sqrt
from ..utils.utils import choose, loglikelihood
import numpy as np
from ..utils.normalization import normalize


def _shrink(p, shrink, weights):
    """
    Helper class for shrink parameters
    :param p: The parameter
    :type p: pyfilter.distributions.continuous.Distribution
    :param shrink: The amount to shrink
    :type shrink: float
    :param weights: The weights to use for estimating the mean
    :type weights: np.ndarray
    :rtype: np.ndarray
    """

    return shrink * p.t_values + (1 - shrink) * np.average(p.t_values, weights=normalize(weights), axis=0)


def _propose(p, indices, h, weights):
    """
    Helper class for shrink parameters
    :param p: The parameter
    :type p: pyfilter.distributions.continuous.Distribution
    :param indices: The indices to resample
    :type indices: np.ndarray
    :param weights: The weights to use for estimating the mean
    :type weights: np.ndarray
    :rtype: np.ndarray
    """
    normalized = normalize(weights)

    means = _shrink(p, sqrt(1 - h ** 2), weights)[indices]
    transformed = p.t_values
    mean = np.average(transformed, weights=normalized, axis=0)
    std = h * np.sqrt(np.average((transformed - mean) ** 2, weights=normalized, axis=0))

    return np.random.normal(means, std, size=p.values.shape)


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
        copy.p_apply(lambda u: _shrink(u, self.a, self._old_w), transformed=True)

        # ===== Weight and get indices ===== #
        t_weights = copy.weight(y, t_x)
        res_ind = self._resamp(t_weights + self._old_w)

        # ===== Propose new parameters ===== #
        self._model.p_apply(lambda u: _propose(u, res_ind, self.h, self._old_w), transformed=True)

        # ===== Propagate the good states ===== #

        resampled_x = choose(self._old_x, res_ind)
        x = self._proposal.draw(y, resampled_x)

        # ===== Calculate the new weights ===== #

        self._old_w = self._proposal.weight(y, x, resampled_x) - choose(t_weights, res_ind)
        self._old_x = x

        self.s_l.append(loglikelihood(self._old_w))
        self.s_mx.append(np.sum(x * normalize(self._old_w), axis=-1))

        if self.saveall:
            self.s_w.append(self._old_w - choose(t_weights, res_ind))
            self.s_x.append(x)

        return self