from .base import BaseKalmanFilter
from ..unscentedtransform import UnscentedTransform
from ..uft import UnscentedFilterTransform
from ..utils import choose
import torch


class UKF(BaseKalmanFilter):
    def __init__(self, model, **kwargs):
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
        (xm, xc), (ym, yc) = self._ut.calc_mean_cov(p)

        xres = torch.empty((steps, *xm.shape))
        yres = torch.empty((steps, *ym.shape))

        xres[0] = xm
        yres[0] = ym

        for i in range(steps - 1):
            spx, spy = self._ut.propagate_sps(xm, xc)
            (xm, xc, _), (ym, yc, _) = self._ut.get_meancov(spx, spy)

            xres[i + 1] = xm
            yres[i + 1] = ym

        return xres, yres