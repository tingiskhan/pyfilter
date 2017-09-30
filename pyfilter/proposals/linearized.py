from .base import Proposal
from ..distributions.continuous import Normal
from numpy import sqrt


def _get_derivs(x, func, h=1e-3):
    """
    EStimates the derivative at x of func.
    :param x:
    :param func:
    :param h:
    :return:
    """

    first = tuple()
    second = tuple()
    for i, tx in enumerate(x):
        up = tx + h
        low = tx - h

        upx, lowx = x.copy(), x.copy()
        upx[i], lowx[i] = up, low

        fupx, flowx = func(upx), func(lowx)

        first += ((fupx - flowx) / 2 / h,)
        second += ((fupx - 2 * func(x) + flowx) / h ** 2,)

    return first, second


class Linearized(Proposal):
    def draw(self, y, x, size=None, *args, **kwargs):
        # TODO: Only works for univariate processes currently
        # ===== Linearize observation density around mean ===== #

        x = [self._meaner(tx) for tx in x]

        t_x = self._model.propagate_apf(x)
        first, second = self._get_derivs(y, t_x, x)

        variances = [-1 / s for s in second]
        mean = [v * f for v, f in zip(variances, first)]

        self._kernel = [Normal(x + m, sqrt(v)) for x, m, v in zip(t_x, mean, variances)]

        return [k.rvs(size=size) for k in self._kernel]

    def _get_derivs(self, y, tx, x):
        """
        Gets the first and second order derivatives.
        :param tx:
        :return:
        """

        return _get_derivs(tx, lambda u: self._model.weight(y, u) + self._model.h_weight(u, x))

    def weight(self, y, xn, xo, *args, **kwargs):
        correction = sum(k.logpdf(_xn) for k, _xn, in zip(self._kernel, xn))
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - correction