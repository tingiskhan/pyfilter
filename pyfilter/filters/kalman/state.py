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
        return self.utf.xm

    def resample(self, indices):
        self.utf.mean.values[:] = choose(self.utf.mean.values, indices)
        self.utf.cov[:] = choose(self.utf.cov, indices)

        self.utf.ym[:] = choose(self.utf.ym, indices)
        self.utf.yc[:] = choose(self.utf.yc, indices)

        self.ll[:] = choose(self.ll, indices)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, indices):
        self.utf.mean.values[indices] = state.utf.mean.values[indices]
        self.utf.cov[indices] = state.utf.cov[indices]

        self.utf.ym[indices] = state.utf.ym[indices]
        self.utf.yc[indices] = state.utf.yc[indices]

        self.ll[indices] = state.ll[indices]
