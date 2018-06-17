from ..timeseries import StateSpaceModel
import numpy as np


class NumericalStateGradient(object):
    h = 1e-6
    grad = None

    def __init__(self, model):
        """
        Implements a way for calculating the gradient and hessian of the underlying states.
        :param model: The model
        :type model: StateSpaceModel
        """

        self._model = model

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

        fmid = self._model.weight(y, x) + self._model.h_weight(x, oldx)

        if self._model.hidden_ndim < 2:
            return 1 / ((self.grad[1] - 2 * fmid + self.grad[0]) / self.h ** 2)

        hess = np.empty((x.shape[0], *x.shape))
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

        return np.linalg.inv(hess.T).T