from .ukf import UKF
from ..utils.unscentedtransform import _get_meancov
from ..utils.utils import customcholesky
from scipy.optimize import minimize, OptimizeResult
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np


class KalmanLaplace(UKF):
    def initialize(self):
        self._initialize_parameters()
        return self

    # TODO: Tidy up and fix stuff

    def _get_x_map(self, y):
        """
        Constructs and performs the MAP optimization of the state variable.
        :return: The optimization results.
        :rtype: OptimizeResult
        """

        if self._old_x is None:
            return minimize(lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x)), self.ssm.hidden.i_mean())

        spx = self._ut.propagate_sps(only_x=True)
        m, c = _get_meancov(spx, self._ut._wm, self._ut._wc)

        if self.ssm.hidden_ndim < 2:
            dist = Normal(m[0], np.sqrt(c[0, 0]))
        else:
            dist = MultivariateNormal(m, customcholesky(c))

        return minimize(lambda x: -(self.ssm.weight(y, x) + dist.logpdf(x)), m)

    def filter(self, y):
        minimized = self._get_x_map(y)

        if self._old_x is None:
            self._ut.initialize(minimized.x)

        self._ut.xmean = self._old_x = minimized.x.copy()
        self._ut.xcov = minimized.hess_inv.copy()

        self.s_mx.append(minimized.x)
        # TODO: Fix this
        self.s_l.append(self.ssm.weight(y, self._old_x))
        self.s_n.append(self._calc_noise(y, self._ut.xmean.copy()))

        return self


class KalmanLaplaceParameters(KalmanLaplace):

    def _opt_params(self, y, x):
        """
        Constructs and performs MAP optimization of the parameters given optimal state.
        :return: The optimization results.
        :rtype: OptimizeResult
        """

    def filter(self, y):
        optstate = self._get_x_map(y)

