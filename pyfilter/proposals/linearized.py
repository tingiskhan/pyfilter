from .base import Proposal
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np
from ..utils.utils import dot


class Linearized(Proposal):
    def draw(self, y, x, size=None, *args, **kwargs):
        x = self._meaner(x)
        t_x = self._model.propagate_apf(x)

        mode, variance = self._get_derivs(y, t_x, x)

        if self._model.hidden.ndim < 2:
            self._kernel = Normal(mode, np.sqrt(variance))
        else:
            self._kernel = MultivariateNormal(mode, np.linalg.cholesky(variance.T).T)

        return self._kernel.rvs(size=size)

    def _get_derivs(self, y, tx, x):
        """
        Gets the first and second order derivatives.
        :param tx:
        :return:
        """

        oldmode = tx
        mode = tx
        converged = False
        iters = 0
        while not converged:
            first, second = self._sg.gradient(y, mode, x), self._sg.hess(y, mode, x)

            if self._model.hidden_ndim < 1:
                mode = mode - first / second
            else:
                inversed = np.linalg.inv(second.T).T
                mode = mode - dot(inversed, first)

            if np.all(np.abs(oldmode - mode) < 1e-2) or iters > 3:
                break

            oldmode = mode.copy()
            iters += 1

        if self._model.hidden_ndim < 2:
            hess = -1 / second
        else:
            hess = -inversed

        return mode, hess

    def weight(self, y, xn, xo, *args, **kwargs):
        correction = self._kernel.logpdf(xn)
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - correction