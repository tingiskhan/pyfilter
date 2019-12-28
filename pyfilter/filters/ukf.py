from .base import BaseKalmanFilter
from ..unscentedtransform import UnscentedTransform
from ..utils import choose


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