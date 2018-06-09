from .base import KalmanFilter
from ..utils.unscentedtransform import UnscentedTransform
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np
from ..utils.utils import customcholesky, choose


class UKF(KalmanFilter):
    def __init__(self, model, *args, utkwargs=None, **kwargs):
        """
        Implements the Unscented Kalman Filter by van der Merwe.
        :param model: The model to use
        :type model: See BaseFilter
        :param args: Any additional arguments
        :type args: See BaseFilter
        :param utkwargs: Any kwargs passed to UnscentedTransform
        :type utkwargs: dict
        :param kwargs: Any additional kwargs passed to `BaseFilter`
        :type kwargs: See BaseFilter
        """
        if 'particles' in kwargs:
            super().__init__(model, *args, **kwargs)
        else:
            super().__init__(model, None, *args, **kwargs)

        self._ut = UnscentedTransform(model, **(utkwargs or {}))

    def initialize(self):
        self._initialize_parameters()
        if self._particles is not None:
            self._ut.initialize(self._model.initialize(size=self._p_particles))
        else:
            self._ut.initialize(self._model.hidden.i_mean())

        return self

    def filter(self, y):
        self._ut.construct(y)

        if self._model.obs_ndim < 2:
            kernel = Normal(self._ut.ymean[0], np.sqrt(self._ut.ycov[0, 0]))
        else:
            kernel = MultivariateNormal(self._ut.ymean, customcholesky(self._ut.ycov))

        x = (self._ut.xmean.copy()[0] if self._model.hidden_ndim < 2 else self._ut.xmean.copy())

        self.s_mx.append(x)
        self.s_l.append(kernel.logpdf(y))

        self.s_n.append(self._calc_noise(y, x))

        return self

    def resample(self, indices):
        self._model.p_apply(lambda x: choose(x[0], indices))

        self._ut._mean = choose(self._ut._mean, indices)
        self._ut._cov = choose(self._ut._cov, indices)
        self.s_l = list(np.array(self.s_l)[:, indices])

        return self