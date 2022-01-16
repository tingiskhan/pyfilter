import torch
from torch import Tensor, zeros_like
from ..state import FilterState, PredictionState
from .unscented.result import UFTCorrectionResult, UFTPredictionResult
from ...utils import choose


class KalmanFilterPrediction(PredictionState):
    """
    Prediction state for particle filters.
    """

    def __init__(
            self,
            prediction: UFTPredictionResult,
            prev_corr: "KalmanFilterState",
            wm: torch.Tensor,
            wc: torch.Tensor
    ):
        """
        Initializes the ``KalmanFilterPrediction`` class.

        Args:
            prediction: The Kalman prediction.
        """

        self.p = prediction
        self.utf = prev_corr.utf
        self.old_ll = prev_corr.get_loglikelihood()
        self.wm = wm
        self.wc = wc

    def create_state_from_prediction(self) -> "FilterState":
        (x_m, x_c), (y_m, y_c) = self.p.get_mean_and_covariance(self.wm, self.wc)
        res = self.utf.update(x_m, x_c, self.p.spx, self.utf, y_m, y_c, self.p.spy)

        return KalmanFilterState(res, zeros_like(self.old_ll))


class KalmanFilterState(FilterState):
    """
    State object for Kalman type filters.
    """

    def __init__(self, utf: UFTCorrectionResult, ll: Tensor):
        """
        Initializes the ``KalmanFilterState`` class.

        Args:
            utf: The correction result.
            ll: The log likelihood.
        """

        super().__init__()
        self.utf = utf
        self.register_buffer("ll", ll)

    def get_mean(self):
        return self.utf.x_dist().mean.clone()

    def get_variance(self):
        return self.utf.x_dist().variance.clone()

    def resample(self, indices):
        self.utf.resample(indices)
        self.ll[:] = choose(self.ll, indices)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, indices):
        self.utf.exchange(indices, state.utf)
        self.ll[indices] = state.ll[indices]

    def get_timeseries_state(self):
        return self.utf.x
