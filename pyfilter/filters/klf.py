from .ukf import UKF
from ..utils.unscentedtransform import _get_meancov
from ..utils.utils import customcholesky
from scipy.optimize import minimize
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np


class KalmanLaplace(UKF):
    def initialize(self):
        self._initialize_parameters()
        return self

    # TODO: Tidy up and fix stuff

    def _get_problem(self, y):
        """
        Constructs the optimization problem.
        :return: Start position and function(s) to minimize.
        :rtype: (np.ndarray, callable)
        """

        if self._old_x is None:
            return self.ssm.hidden.i_mean(), lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x))

        spx = self._ut.propagate_sps(only_x=True)
        m, c = _get_meancov(spx, self._ut._wm, self._ut._wc)

        if self.ssm.hidden_ndim < 2:
            dist = Normal(m[0], np.sqrt(c[0, 0]))
        else:
            dist = MultivariateNormal(m, customcholesky(c))

        return m, lambda x: -(self.ssm.weight(y, x) + dist.logpdf(x))

    def filter(self, y):
        start, func = self._get_problem(y)

        minimzed = minimize(func, start)

        if self._old_x is None:
            self._ut.initialize(minimzed.x)

        self._ut.xmean = self._old_x = minimzed.x.copy()
        self._ut.xcov = minimzed.hess_inv.copy()

        self.s_mx.append(minimzed.x)
        # TODO: Fix this
        self.s_l.append(self.ssm.weight(y, self._old_x))
        self.s_n.append(self._calc_noise(y, self._ut.xmean.copy()))

        return self