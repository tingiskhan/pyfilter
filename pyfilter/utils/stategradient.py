from ..timeseries import StateSpaceModel
import autograd as ag
import numpy as np


class StateGradient(object):
    def __init__(self, model):
        """
        Implements a way for calculating the gradient and hessian of the underlying states.
        :param model: The model
        :type model: StateSpaceModel
        """

        self._model = model

    @property
    def _ograd(self):
        return ag.elementwise_grad(self._model.weight, argnum=1)

    @property
    def _hgrad(self):
        return ag.elementwise_grad(self._model.h_weight, argnum=0)

    @property
    def _ohess(self):
        return ag.elementwise_grad(self._ograd, argnum=1)

    @property
    def _hhess(self):
        return ag.elementwise_grad(self._hgrad, argnum=0)

    def gradient(self, y, x, oldx):
        """
        Calculates the gradient of both the hidden and observable state.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ograd(y, x) + self._hgrad(x, oldx)

    def hess(self, y, x, oldx):
        """
        Calculates the hessian of both the hidden and observable state.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        if self._model.hidden_ndim < 2:
            return self._ohess(y, x) + self._hhess(x, oldx)

        hess = np.zeros((x.shape[0], *x.shape))
        inds = np.diag_indices(x.shape[0])

        hess[inds] = self._ohess(y, x) + self._hhess(x, oldx)

        return hess

    def o_gradient(self, y, x, oldx):
        """
        Calculates the gradient of the function at x.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ograd(y, x)

    def o_hess(self, y, x, oldx):
        """
        Calculates the hessian of the function at x.
        Calculates the gradient of the function at x.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ohess(y, x)


class NumericalStateGradient(StateGradient):
    h = 1e-6
    grad = None

    def gradient(self, y, x, oldx):
        """
        Estimates the gradient numerically.
        :param y: The observation
        :param x: The state
        :param oldx: The previous state
        :return:
        """

        if self._model.hidden_ndim < 2:
            up = x + self.h
            low = x - self.h

            fupx = self._model.weight(y, up) + self._model.h_weight(up, oldx)
            flowx = self._model.weight(y, low) + self._model.h_weight(low, oldx)

            self.grad = [flowx, fupx]

            return (fupx - flowx) / 2 / self.h

        grad = np.empty_like(x)
        self.grad = list()
        for i, tx in enumerate(x):
            up, low = x.copy(), x.copy()
            up[i] = tx + self.h
            low[i] = tx - self.h

            fupx = self._model.weight(y, up) + self._model.h_weight(up, oldx)
            flowx = self._model.weight(y, low) + self._model.h_weight(low, oldx)

            grad[i] = (fupx - flowx) / 2 / self.h
            self.grad.append((fupx, flowx))

        return grad

    def hess(self, y, x, oldx):
        """
        Estimates the hessian numerically.
        :param y: The observation
        :param x: The state
        :param oldx: The previous state
        :return:
        """

        fx = self._model.weight(y, x) + self._model.h_weight(x, oldx)

        if self._model.hidden_ndim < 2:
            return (self.grad[1] - 2 * fx + self.grad[0]) / self.h ** 2

        hess = np.empty((x.shape[0], *x.shape))
        fmid = self._model.weight(y, x) + self._model.h_weight(x, oldx)
        for i in range(x.shape[0]):
            for j in range(i, x.shape[0]):
                if i == j:
                    hess[i, i] = (self.grad[i][0] - 2 * fmid + self.grad[i][1]) / self.h ** 2
                else:
                    upx, lowx = x.copy(), x.copy()

                    upx[i] += self.h
                    upx[j] += self.h

                    lowx[i] -= self.h
                    lowx[j] -= self.h

                    fup = self._model.weight(y, upx) + self._model.h_weight(upx, oldx)
                    flow = self._model.weight(y, lowx) + self._model.h_weight(lowx, oldx)

                    tmp = fup - self.grad[i][0] - self.grad[j][0] + 2 * fmid - self.grad[i][1] - self.grad[j][1] + flow

                    hess[i, j] = hess[j, i] = tmp / 2 / self.h ** 2

        return hess