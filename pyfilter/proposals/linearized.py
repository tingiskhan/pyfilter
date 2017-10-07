from .base import Proposal
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np
from ..utils.utils import dot


def _get_derivs(x, func, ndim, h=1e-3):
    """
    EStimates the derivative at x of func.
    :param x:
    :param func:
    :param h:
    :return:
    """

    if ndim < 2:

        up = x + h
        low = x - h

        fupx, flowx = func(up), func(low)

        first = (fupx - flowx) / 2 / h
        second = (fupx - 2 * func(x) + flowx) / h ** 2

    else:
        first = np.empty_like(x)
        second = np.zeros((x.shape[0], *x.shape))

        for i, tx in enumerate(x):
            up, low = x.copy(), x.copy()
            up[i] = tx + h
            low[i] = tx - h

            fupx, flowx = func(up), func(low)

            first[i] = (fupx - flowx) / 2 / h
            second[i, i] = (fupx - 2 * func(x) + flowx) / h ** 2

    return first, second


class Linearized(Proposal):
    def draw(self, y, x, size=None, *args, **kwargs):
        x = self._meaner(x)
        t_x = self._model.propagate_apf(x)

        first, second = self._get_derivs(y, t_x, x)

        if self._model.hidden.ndim < 2:
            variances = -1 / second
            mean = variances * first
            self._kernel = Normal(x + mean, np.sqrt(variances))
        else:
            variances = -np.linalg.inv(second.T).T
            mean = dot(variances, first)
            self._kernel = MultivariateNormal(x + mean, np.linalg.cholesky(variances.T).T)

        return self._kernel.rvs(size=size)

    def _get_derivs(self, y, tx, x):
        """
        Gets the first and second order derivatives.
        :param tx:
        :return:
        """

        return _get_derivs(tx, lambda u: self._model.weight(y, u) + self._model.h_weight(u, x), self._model.hidden.ndim)

    def weight(self, y, xn, xo, *args, **kwargs):
        correction = self._kernel.logpdf(xn)
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - correction