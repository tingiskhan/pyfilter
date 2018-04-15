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
    def filter(self, y):
        if self._old_x is None:
            func = lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x))
            start = self.ssm.hidden.i_mean()
        else:
            spx = self._ut.propagate_sps(only_x=True)

            m, c = _get_meancov(spx, self._ut._wm, self._ut._wc)

            if self.ssm.hidden_ndim < 2:
                dist = Normal(m, np.sqrt(c))
            else:
                dist = MultivariateNormal(m, customcholesky(c))

            func = lambda x: -(self.ssm.weight(y, x) + dist.logpdf(x))
            start = m

        minimzed = minimize(func, start)

        if self._old_x is None:
            self._ut.initialize(minimzed.x)

        self._ut.xmean = self._old_x = minimzed.x.copy()
        self._ut.xcov = minimzed.hess_inv.copy()

        self.s_mx.append(minimzed.x)
        # TODO: Fix this
        # self.s_l.append(kernel.logpdf(y))

        self.s_n.append(self._calc_noise(y, self._ut.xmean.copy()))

        return self