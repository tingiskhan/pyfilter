from .ukf import UKF
from ..utils.unscentedtransform import _get_meancov
from ..utils.utils import customcholesky, bfgs
from scipy.optimize import minimize, OptimizeResult
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np
from scipy import stats


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
            start = self.ssm.hidden.i_mean() if self._particles is None else self.ssm.initialize(size=self._p_particles)

            if self.ssm.hidden.ndim < 2 and isinstance(start, np.ndarray):
                start = start[None]

            return bfgs(lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x)), start)

        spx = self._ut.propagate_sps(only_x=True)
        m, c = _get_meancov(spx, self._ut._wm, self._ut._wc)

        if self.ssm.hidden_ndim < 2:
            dist = Normal(m[0], np.sqrt(c[0, 0]))
        else:
            dist = MultivariateNormal(m, customcholesky(c))

        return bfgs(lambda x: -(self.ssm.weight(y, x) + dist.logpdf(x)), m)

    def _save(self, y, optstate):
        """
        Saves the data.
        :param optstate: The optimal state
        :return: Self
        :rtype: KalmanLaplace
        """

        if self._old_x is None:
            self._ut.initialize(optstate.x)

        self._ut.xmean = self._old_x = optstate.x.copy()
        self._ut.xcov = optstate.hess_inv.copy()

        self.s_mx.append(optstate.x)
        # TODO: Fix this
        self.s_l.append(self.ssm.weight(y, self._old_x))
        self.s_n.append(self._calc_noise(y, self._ut.xmean.copy()))

        return self

    def filter(self, y):
        optstate = self._get_x_map(y)

        return self._save(y, optstate)