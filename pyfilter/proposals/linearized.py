from .base import Proposal
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np
from ..utils.utils import dot, customcholesky


class Linearized(Proposal):
    def draw(self, y, x, size=None, *args, **kwargs):
        x = self._meaner(x)
        t_x = self._model.propagate_apf(x)

        mode, variance = self._get_mode_variance(y, t_x, x)

        if self._model.hidden.ndim < 2:
            self._kernel = Normal(mode, np.sqrt(variance))
        else:
            self._kernel = MultivariateNormal(mode, customcholesky(variance))

        return self._kernel.rvs(size=size)

    def _get_mode_variance(self, y, tx, x):
        """
        Gets the first and second order derivatives.
        :param tx:
        :return:
        """

        mode = tx
        converged = False
        iters = 0
        hess = None
        while not converged:
            first, hess = self._sg.gradient(y, mode, x), self._sg.hess(y, mode, x)

            if self._model.hidden_ndim < 2:
                mode = mode - hess * first
            else:
                mode = mode - dot(hess, first)

            if np.all(np.abs(first) < 1e-4) or iters > 3:
                break

            iters += 1

        return mode, -hess

    def weight(self, y, xn, xo, *args, **kwargs):
        correction = self._kernel.logpdf(xn)
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - correction