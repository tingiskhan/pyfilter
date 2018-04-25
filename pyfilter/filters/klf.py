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
            return bfgs(lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x)), self.ssm.hidden.i_mean())

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


def _define_pdf(params, scale=1e-3):
    """
    Helper function for creating the PDF.
    :param params: The parameters to use for defining the distribution
    :type params: (np.ndarray, Distribution)
    :return: A truncated normal distribution
    :rtype: stats.truncnorm
    """

    mean = params[0]
    std = params[1].std() * scale

    a = (params[1].bounds()[0] - mean) / std
    b = (params[1].bounds()[1] - mean) / std

    return stats.truncnorm(a, b, mean, std)


class KalmanLaplaceParameters(KalmanLaplace):
    """
    Implements a Kalman-Laplace filter targeting the parameters as well as the state, using artificial dynamics for the
    parameters.
    """
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

        hdynamics, odynamics = self.ssm.p_map(_define_pdf)

        def s_func(p):
            copied = self._get_copy(p)

            dynprob = sum(d.logpdf(px) for d, px in zip(odynamics + hdynamics, p.tolist()))

            return -(copied.weight(y, x) + copied.h_weight(x, self._old_x) + dynprob)

        return minimize(s_func, ostart + hstart, bounds=obsbounds + hidbounds)

    def filter(self, y):
        optstate = self._get_x_map(y)
        params = self._get_p_map(y, optstate.x)

        self.ssm.observable.theta = tuple(params.x[:len(self.ssm.ind_obsparams)])
        self.ssm.hidden.theta = tuple(params.x[len(self.ssm.ind_obsparams):])

        return self._save(y, optstate)

