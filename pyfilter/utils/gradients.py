from ..model import StateSpaceModel
import numpy as np
from .normalization import normalize


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

    def _initialize(self):
        """
        :rtype: tuple of list
        """
        hiddens = list()
        for h in self._model.hidden:
            hiddens.append(len(h.theta) * [0])

        return hiddens, len(self._model.observable.theta) * [0]

    def get_gradient(self, y, x, xo, w):
        """
        Estimates the gradient at the current values of the parameters and current states.
        :param y: The current observation
        :param x: The current state
        :type x: list of numpy.ndarray
        :param xo: The previous state
        :type xo: list of numpy.ndarray
        :param w: The weights
        :type w: np.ndarray
        :return:
        """

        hgrads, ograds = self._model.p_grad(y, x, xo, h=self._h)
        normalized = normalize(w)

        self._get_hidden_gradients(hgrads, normalized)._get_obs_gradients(ograds, normalized)

        return self

    def _get_hidden_gradients(self, hgrads, normalized):
        """
        Helper method.
        :return:
        """

        for i, (hgrad, oldhgrad, hm) in enumerate(zip(hgrads, self.oldgrad[0], self._m[0])):
            for j, (hg, ohg, m) in enumerate(zip(hgrad, oldhgrad, hm)):
                newm = self._shrink * m + (1 - self._shrink) * ohg + hg
                newgrad = np.sum(normalized * newm, axis=-1)

                self._m[0][i][j] = newm
                self.oldgrad[0][i][j] = newgrad

        return self

    def _get_obs_gradients(self, ograds, normalized):
        """
        Helper method
        :return:
        """

        for k, (og, oog, m) in enumerate(zip(ograds, self.oldgrad[1], self._m[1])):
            newm = self._shrink * m + (1 - self._shrink) * oog + og
            newgrad = np.sum(normalized * newm, axis=-1)

            self._m[1][k] = newm
            self.oldgrad[1][k] = newgrad

        return self