from .base import BaseFilter
from ..distributions.continuous import Normal
from ..helpers.resampling import systematic
import numpy as np
from ..helpers.helpers import choose, loglikelihood


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


class Linearized(BaseFilter):
    def filter(self, y):
        # TODO: Only works for univariate processes currently
        # ===== Linearize observation density around mean ===== #
        t_x = self._model.propagate_apf(self._old_x)
        first, second = self._get_derivs(y, t_x)

        variances = [-1 / s for s in second]
        mean = [v * f for v, f in zip(variances, first)]

        # ===== Define kernels ===== #

        kernels = [Normal(x + m, np.sqrt(v)) for x, m, v in zip(self._old_x, mean, variances)]
        newx = [kernel.rvs() for kernel in kernels]

        # ===== Weight and model ===== #

        modw = self._model.weight(y, newx) + self._model.h_weight(newx, self._old_x)
        weight = modw - sum(kernel.logpdf(x) for kernel, x in zip(kernels, newx))

        inds = systematic(weight)

        self._old_x = choose(newx, inds)
        self._old_w = weight
        self.s_l.append(loglikelihood(weight))

        if self.saveall:
            self.s_x.append(newx)
            self.s_w.append(weight)

        return self

    def _get_derivs(self, y, tx):
        """
        Gets the first and second order derivatives.
        :param tx:
        :return:
        """

        return _get_derivs(tx, lambda u: self._model.weight(y, u) + self._model.h_weight(u, self._old_x))