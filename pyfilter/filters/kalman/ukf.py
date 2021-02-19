import torch
from typing import Dict
from .base import BaseKalmanFilter
from .unscented import UnscentedFilterTransform
from .state import KalmanState


class UKF(BaseKalmanFilter):
    def __init__(self, model, utfkwargs: Dict[str, object] = None):
        """
        Implements the Unscented Kalman Filter by van der Merwe.
        :param utfkwargs: Any kwargs passed to `UnscentedFilterTransform`
        """

        super().__init__(model)
        self._ut = UnscentedFilterTransform(model, **(utfkwargs or dict()))

    def initialize(self) -> KalmanState:
        res = self._ut.initialize(self.n_parallel)
        return KalmanState(res, torch.zeros(self.n_parallel, device=res.xm.device))

    def _filter(self, y, state: KalmanState):
        p = self._ut.predict(state.utf)
        res = self._ut.correct(y, p, state.utf)

        return KalmanState(res, res.y_dist().log_prob(y))

    def predict(self, state: KalmanState, steps, *args, **kwargs):
        utf_state = state.utf

        p = self._ut.predict(state.utf)
        c = self._ut.calc_mean_cov(p)

        utf_state = self._ut.update_state(c.xm, c.xc, p.spx, utf_state)

        xres = torch.empty((steps, *c.xm.shape))
        yres = torch.empty((steps, *c.ym.shape))

        xres[0] = c.xm
        yres[0] = c.ym

        for i in range(steps - 1):
            p = self._ut.predict(utf_state)
            c = self._ut.calc_mean_cov(p)

            utf_state = self._ut.update_state(c.xm, c.xc, p.spx, utf_state)

            xres[i + 1] = c.xm
            yres[i + 1] = c.ym

        return xres, yres
