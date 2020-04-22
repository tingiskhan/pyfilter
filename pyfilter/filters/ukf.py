from .base import BaseKalmanFilter
from ..unscentedtransform import UnscentedTransform
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

        self._ut = UnscentedTransform(model, **kwargs)

    def initialize(self):
        self._ut.initialize(self._model.hidden.i_sample(self._n_parallel))

        return self

    def _filter(self, y):
        self._ut = self._ut.construct(y)    # type: UnscentedTransform

        return self._ut.xmean, self._ut.y_dist.log_prob(y)

    def _resample(self, inds):
        self._ut.xmean = choose(self._ut.xmean, inds)
        self._ut.xcov = choose(self._ut.xcov, inds)

        return self

    def predict(self, steps, *args, **kwargs):
        spx, spy = self._ut.propagate_sps()
        (xm, xc, _), (ym, yc, _) = self._ut.get_meancov(spx, spy)

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