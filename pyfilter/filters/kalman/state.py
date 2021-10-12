from torch import Tensor
from ..state import BaseFilterState
from .unscented.result import UFTCorrectionResult
from ...utils import choose


class KalmanFilterState(BaseFilterState):
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
