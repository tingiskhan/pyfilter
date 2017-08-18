from .base import BaseFilter
from math import sqrt
from ..helpers.helpers import choose, loglikelihood
import scipy.stats as stats
import numpy as np
from ..helpers.normalization import normalize


def _shrink(parameter, shrink):
    """
    Helper class for shrink parameters
    :param parameter: 
    :param shrink: 
    :return: 
    """

    return shrink * parameter + (1 - shrink) * parameter.mean()


def _propose(bounds, params, indices, h, particles, weights):
    """
    Helper class for proposing a parameter
    :param bounds:
    :param params:
    :param indices:
    :param weights:
    :return:
    """

    normalized = normalize(weights)

    means = _shrink(params[indices], sqrt(1 - h ** 2))
    std = h * np.sqrt(np.average((params - params.mean()) ** 2, weights=normalized, axis=0))

    # TODO: Check the truncnorm - doesn't seem to work
    a = (bounds[0] - means) / std
    b = (bounds[1] - means) / std

    return stats.truncnorm.rvs(a, b, means, std, size=particles)


class RAPF(BaseFilter):

    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def _shrink(self):

        # ===== HIDDEN ===== #

        for i, ts in enumerate(self._model.hidden):
            parameters = tuple()
            for j, p in enumerate(ts.theta):
                if j in self._h_params[i].keys():
                    parameters += (_shrink(p, self.a),)
                else:
                    parameters += (p,)

            self._copy.hidden[i].theta = parameters

        # ===== OBSERVABLE ===== #

        parameters = tuple()
        for j, p in enumerate(self._model.observable.theta):
            if j in self._o_params.keys():
                parameters += (_shrink(p, self.a),)
            else:
                parameters += (p,)

        self._copy.observable.theta = parameters

        return self

    def _propagate_parameters(self, indices, weights):
        # ===== HIDDEN ===== #
        for i, ts in enumerate(self._model.hidden):
            parameters = tuple()
            for j, p in enumerate(ts.theta):
                if j in self._h_params[i].keys():
                    parameters += (_propose(self._h_params[i][j], p, indices, self.h, self._p_particles, weights),)
                else:
                    parameters += (p,)

            ts.theta = parameters

        # ===== OBSERVABLE ===== #
        parameters = tuple()
        for j, p in enumerate(self._model.observable.theta):
            if j in self._o_params.keys():
                parameters += (_propose(self._o_params[j], p, indices, self.h, self._p_particles, weights),)
            else:
                parameters += (p,)

        self._model.observable.theta = parameters

        return self

    def filter(self, y):
        # TODO: Implement a way of sampling using other parameters
        self._shrink()
        t_x = self._copy.propagate_apf(self._old_x)
        t_weights = self._copy.weight(y, t_x)

        try:
            resampled_indices = self._resamp(t_weights + self.s_w[-1])
        except IndexError:
            resampled_indices = self._resamp(t_weights)

        self._propagate_parameters(resampled_indices, t_weights)

        resampled_x = choose(self._old_x, resampled_indices)
        x = self._proposal.draw(y, resampled_x)
        weights = self._proposal.weight(y, x, resampled_x)

        self._old_x = x
        self._old_y = y

        self.s_w.append(weights - choose(t_weights, resampled_indices))
        self.s_x.append(x)
        self.s_l.append(loglikelihood(weights))

        return self