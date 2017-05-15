from .base import BaseFilter
from ..distributions.continuous import Distribution
import numpy as np
from ..helpers.normalization import normalize
from math import sqrt
from ..helpers.resampling import systematic
from ..helpers.helpers import choose
import scipy.stats as stats


def _propose(bounds, params, indices, h, particles):
    """
    Helper class for proposing a parameter
    :param bounds:
    :param params:
    :param indices:
    :return:
    """

    means = params[indices]
    std = h * params.std()

    # TODO: Check the truncnorm - doesn't seem to work
    a = (bounds[0] - means) / std
    b = (bounds[1] - means) / std

    return stats.truncnorm.rvs(a, b, means, std, size=particles)


class RAPF(BaseFilter):

    def __init__(self, model, particles, *args, shrinkage=0.95, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self.a = (3 * shrinkage - 1) / 2 / shrinkage
        self.h = sqrt(1 - self.a ** 2)

    def _initialize_parameters(self):

        # ===== HIDDEN ===== #

        self._h_params = list()
        for i, ts in enumerate(self._model.hidden):
            temp = dict()
            parameters = tuple()
            for j, p in enumerate(ts.theta):
                if isinstance(p, Distribution):
                    temp[j] = p.bounds()
                    parameters += (p.rvs(size=self._particles),)
                else:
                    parameters += (p,)

            ts.theta = parameters
            self._h_params.append(temp)

        # ===== OBSERVABLE ===== #

        self._o_params = dict()
        parameters = tuple()
        for j, p in enumerate(self._model.observable.theta):
            if isinstance(p, Distribution):
                self._o_params[j] = p.bounds()
                parameters += (p.rvs(size=self._particles),)
            else:
                parameters += (p,)

        self._model.observable.theta = parameters

        return self

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)

        return self

    def _shrink(self):

        # ===== HIDDEN ===== #

        for i, ts in enumerate(self._model.hidden):
            parameters = tuple()
            for j, p in enumerate(ts.theta):
                if j in self._h_params[i].keys():
                    parameters += (self.a * p + (1 - self.a) * p.mean(),)
                else:
                    parameters += (p,)

            ts.theta = parameters

        # ===== OBSERVABLE ===== #

        parameters = tuple()
        for j, p in enumerate(self._model.observable.theta):
            if j in self._o_params.keys():
                parameters += (self.a * p + (1 - self.a) * p.mean(),)
            else:
                parameters += (p,)

        self._model.observable.theta = parameters

        return self

    def _propagate_parameters(self, indices):
        # ===== HIDDEN ===== #
        for i, ts in enumerate(self._model.hidden):
            parameters = tuple()
            for j, p in enumerate(ts.theta):
                if j in self._h_params[i].keys():
                    parameters += (_propose(self._h_params[i][j], p, indices, self.h, self._particles),)
                else:
                    parameters += (p,)

            ts.theta = parameters

        # ===== OBSERVABLE ===== #
        parameters = tuple()
        for j, p in enumerate(self._model.observable.theta):
            if j in self._o_params.keys():
                parameters += (_propose(self._o_params[j], p, indices, self.h, self._particles),)
            else:
                parameters += (p,)

        self._model.observable.theta = parameters

        return self

    def filter(self, y):
        self._shrink()
        t_x = self._model.propagate_apf(self._old_x)
        t_weights = self._model.weight(y, t_x)

        try:
            resampled_indices = systematic(t_weights + self.s_w[-1])
        except IndexError:
            resampled_indices = systematic(t_weights)

        self._propagate_parameters(resampled_indices)
        x = self._model.propagate(choose(self._old_x, resampled_indices))
        weights = self._model.weight(y, x)

        self._old_x = x
        self._old_y = y

        self.s_w.append(weights - choose(t_weights, resampled_indices))
        self.s_x.append(x)
        self.s_l.append(weights.mean(axis=-1))

        return self