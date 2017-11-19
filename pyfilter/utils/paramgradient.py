import numpy as np
from pyfilter.timeseries import StateSpaceModel
from .normalization import normalize
from .utils import choose


def _calc_grad_and_m(shrinkage, oldm, oldgrad, currgrad, weights, indices):
    try:
        newm = shrinkage * choose(oldm, indices) + (1 - shrinkage) * oldgrad + currgrad
    except AttributeError:
        newm = (1 - shrinkage) * oldgrad + currgrad

    newgrad = np.sum(weights * newm, axis=-1)

    return newm, newgrad


class GradientEstimator(object):
    def __init__(self, model, shrinkage=0.95, h=1e-3):
        """
        Helper class for estimating the parameter gradients of `model`.
        :param model: The model to estimate the gradient for.
        :type model: StateSpaceModel
        """

        self._model = model
        self._h = h
        self._m = self._initialize()
        self._shrink = shrinkage
        self.oldgrad = self._initialize()
        self.diffgrad = self._initialize()

    def _initialize(self):
        """
        :rtype: tuple of list
        """

        return len(self._model.hidden.theta) * [0], len(self._model.observable.theta) * [0]

    def update_gradient(self, y, x, xo, w, inds):
        """
        Estimates the gradient at the current values of the parameters and current states.
        :param y: The current observation
        :param x: The current state
        :type x: list of numpy.ndarray
        :param xo: The previous state
        :type xo: list of numpy.ndarray
        :param w: The weights
        :type w: np.ndarray
        :param inds: The indices
        :type inds: np.ndarray
        :return:
        """

        hgrads, ograds = self._model.p_grad(y, x, xo, h=self._h)
        normalized = normalize(w)

        self._get_hidden_gradients(hgrads, normalized, inds)._get_obs_gradients(ograds, normalized, inds)

        return self

    def _get_hidden_gradients(self, hgrads, normalized, inds):
        """
        Helper method.
        :return:
        """

        for i, (grad, oldgrad, oldm) in enumerate(zip(hgrads, self.oldgrad[0], self._m[0])):
            newm, newgrad = _calc_grad_and_m(self._shrink, oldm, oldgrad, grad, normalized, inds)

            self._m[0][i] = newm
            self.diffgrad[0][i] = newgrad - self.oldgrad[0][i]
            self.oldgrad[0][i] = newgrad

        return self

    def _get_obs_gradients(self, ograds, normalized, inds):
        """
        Helper method
        :return:
        """

        for k, (grad, oldgrad, oldm) in enumerate(zip(ograds, self.oldgrad[1], self._m[1])):
            newm, newgrad = _calc_grad_and_m(self._shrink, oldm, oldgrad, grad, normalized, inds)

            self._m[1][k] = newm
            self.diffgrad[1][k] = newgrad - self.oldgrad[1][k]
            self.oldgrad[1][k] = newgrad

        return self