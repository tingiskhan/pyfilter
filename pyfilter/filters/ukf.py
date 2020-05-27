from .base import BaseKalmanFilter
from ..uft import UnscentedFilterTransform, UFTCorrectionResult
from ..utils import choose
import torch


class UKF(BaseKalmanFilter):
    def __init__(self, model, **kwargs):
        """
        Implements the Unscented Kalman Filter by van der Merwe.
        :param kwargs: Any kwargs passed to UnscentedTransform
        """

        super().__init__(model)

        self._ut = UnscentedFilterTransform(model, **kwargs)
        self._ut_res = None

    def initialize(self):
        self._ut_res = self._ut.initialize(self._n_parallel)

        return self

    def _filter(self, y):
        p = self._ut.predict(self._ut_res)
        self._ut_res = self._ut.correct(y, p)

        return self._ut_res.xm, self._ut_res.y_dist().log_prob(y)

    def _resample(self, inds):
        self._ut_res.xm = choose(self._ut_res.xm, inds)
        self._ut_res.xc = choose(self._ut_res.xc, inds)

        return self

    def predict(self, steps, *args, **kwargs):
        p = self._ut.predict(self._ut_res)
        c = self._ut.calc_mean_cov(p)

        xres = torch.empty((steps, *c.xm.shape))
        yres = torch.empty((steps, *c.ym.shape))

        xres[0] = c.xm
        yres[0] = c.ym

        for i in range(steps - 1):
            p = self._ut.predict(c)
            c = self._ut.calc_mean_cov(p)

            xres[i + 1] = c.xm
            yres[i + 1] = c.ym

        return xres, yres