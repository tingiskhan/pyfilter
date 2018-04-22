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


class KalmanLaplaceParameters(KalmanLaplace):
    def _params(self, x):
        """
        Constructs the parameter space
        :param x:
        :return:
        """
        obsp = np.array(self.ssm.observable.theta)
        hidp = np.array(self.ssm.hidden.theta)

        obsp[self.ssm.ind_obsparams] = x[:len(self.ssm.ind_obsparams)]
        hidp[self.ssm.ind_hiddenparams] = x[len(self.ssm.ind_obsparams):]

        return obsp, hidp

    def _get_copy(self, p):
        """
        Copies current state and overwrites parameters with p.
        :param p: The paramters to use
        :type p: np.ndarray
        :return: Copy of current SSM
        :rtype: pyfilter.timeseries.model.StateSpaceModel
        """

        copied = self.ssm.copy()

        obsp, hidp = self._params(p)

        copied.hidden.theta = tuple(hidp.tolist())
        copied.observable.theta = tuple(obsp.tolist())

        return copied

    def _get_p_map(self, y, x):
        """
        Constructs and performs MAP optimization of the parameters given optimal state.
        :return: The optimization results.
        :rtype: OptimizeResult
        """

        ostart = np.array(self.ssm.observable.theta)[self.ssm.ind_obsparams].tolist()
        hstart = np.array(self.ssm.hidden.theta)[self.ssm.ind_hiddenparams].tolist()

        obsbounds, hidbounds = self.ssm.optbounds

        if self._old_x is None:
            def i_func(p):
                copied = self._get_copy(p)
                return -(copied.weight(y, x) + copied.hidden.i_weight(x) + copied.p_prior())

            return minimize(i_func, ostart + hstart, bounds=obsbounds + hidbounds)

        # TODO: Fix such that we can perform online optimization. Requires defining artificial dynamics for parameters

        return

    def filter(self, y):
        optstate = self._get_x_map(y)
        params = self._get_p_map(y, optstate.x)

        # TODO: Fix overwriting of parameters

        return self._save(y, optstate)

