from torch import Tensor
from ..state import BaseState
from .unscented.result import UFTCorrectionResult
from ...utils import choose


class KalmanState(BaseState):
    def __init__(self, utf: UFTCorrectionResult, ll: Tensor):
        super().__init__()
        self.utf = utf
        self.register_buffer("ll", ll)

    def get_mean(self):
        return self.utf.x_dist().mean

    # TODO: Fix this
    def resample(self, indices):
        self.utf.resample(indices)
        self.ll[:] = choose(self.ll, indices)

    def get_loglikelihood(self):
        return self.ll

    # TODO: Fix this
    def exchange(self, state, indices):
        self.utf.exchange(indices, state.utf)
        self.ll[indices] = state.ll[indices]

    def get_timeseries_state(self):
        return self.utf.x
