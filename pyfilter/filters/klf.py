from .ukf import UKF
from ..utils.utils import customcholesky, bfgs
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np
from scipy.optimize import minimize


class KalmanLaplace(UKF):
    def __init__(self, model, *args, **kwargs):
        """
        Implements the Kalman-Laplace filter engineered by Paul Bui Qang and Christian Musso. Found here:
            https://ieeexplore.ieee.org/abstract/document/7266743/
        :param model: See Base
        :param args: See UKF
        :param kwargs: See UKF
        """

        super().__init__(model, *args, **kwargs)

        self._opt = None

    def initialize(self):
        self._initialize_parameters()
        return self

    # TODO: Tidy up and fix stuff

    def _get_x_map(self, y):
        """
        Constructs and performs the MAP optimization of the state variable.
        :return: The optimization results.
        :rtype: pyfilter.utils.utils.OptimizeResult
        """

        if self._old_x is None:
            start = self.ssm.hidden.i_mean() if self._particles is None else self.ssm.initialize(size=self._p_particles)

            self._opt = minimize if self._particles is None else bfgs

            if self.ssm.hidden.ndim < 2 and isinstance(start, np.ndarray):
                start = start[None]

            return self._opt(lambda x: -(self.ssm.weight(y, x) + self.ssm.hidden.i_weight(x)), start)

        (m, c, _), _ = self._ut.get_meancov()

        if self.ssm.hidden_ndim < 2:
            dist = Normal(m[0], np.sqrt(c[0, 0]))
        else:
            dist = MultivariateNormal(m, customcholesky(c))

        return self._opt(lambda x: -(self.ssm.weight(y, x) + dist.logpdf(x)), m)

    def _save(self, y, optstate):
        """
        Saves the data.
        :param optstate: The optimal state
        :return: Self
        :rtype: KalmanLaplace
        """

        if self._old_x is None:
            self._ut.initialize(optstate.x if self.ssm.hidden_ndim > 1 else optstate.x[0])

        # TODO: Speed this up since we already do the computations earlier
        _, (ym, yc, _) = self._ut.get_meancov()

        if self.ssm.obs_ndim < 2:
            dist = Normal(ym[0], np.sqrt(yc[0, 0]))
        else:
            dist = MultivariateNormal(ym, customcholesky(yc))

        self._ut.xmean = self._old_x = optstate.x
        self._ut.xcov = optstate.hess_inv

        self.s_mx.append(optstate.x)
        # TODO: Fix this
        # TODO: Investigate the discrepancy between the SISR and this filter in the log likelihood estimation
        self.s_l.append(dist.logpdf(y))
        self.s_n.append(self._calc_noise(y, self._ut.xmean.copy()))

        return self

    def filter(self, y):
        optstate = self._get_x_map(y)

        return self._save(y, optstate)