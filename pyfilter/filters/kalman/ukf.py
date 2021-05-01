import torch
from typing import Dict
from .base import BaseKalmanFilter
from .unscented import UnscentedFilterTransform
from .state import KalmanState


class UKF(BaseKalmanFilter):
    def __init__(self, model, utf_kwargs: Dict[str, object] = None):
        """
        Implements the Unscented Kalman Filter by van der Merwe.

        :param utf_kwargs: Any kwargs passed to `UnscentedFilterTransform`
        """

        super().__init__(model)
        self._ut = UnscentedFilterTransform(model, **(utf_kwargs or dict()))

    def initialize(self) -> KalmanState:
        res = self._ut.initialize(self.n_parallel)
        return KalmanState(res, torch.zeros(self.n_parallel, device=res.x.device))

    def predict_correct(self, y, state: KalmanState):
        p = self._ut.predict(state.utf)

        if torch.isnan(y).any():
            (x_m, x_c), (y_m, y_c) = p.get_mean_and_covariance(self._ut._wm, self._ut._wc)
            res = self._ut.update_state(x_m, x_c, p.spx, state.utf, y_m, y_c, p.spy)

            return KalmanState(res, torch.zeros_like(state.ll))

        res = self._ut.correct(y, p, state.utf)

        return KalmanState(res, res.y.dist.log_prob(y))

    def predict(self, state: KalmanState, steps, *args, **kwargs):
        utf_state = state.utf

        p = self._ut.predict(state.utf)
        (x_m, x_c), (y_m, y_c) = p.get_mean_and_covariance(self._ut._wm, self._ut._wc)

        utf_state = self._ut.update_state(x_m, x_c, p.spx, utf_state, y_m, y_c, p.spy)

        x_res = torch.empty((steps, *x_m.shape))
        y_res = torch.empty((steps, *y_m.shape))

        x_res[0] = x_m
        y_res[0] = y_m

        for i in range(steps - 1):
            p = self._ut.predict(utf_state)
            (x_m, x_c), (y_m, y_c) = p.get_mean_and_covariance(self._ut._wm, self._ut._wc)

            utf_state = self._ut.update_state(x_m, x_c, p.spx, utf_state, y_m, y_c, p.spy)

            x_res[i + 1] = x_m
            y_res[i + 1] = y_m

        return x_res, y_res
