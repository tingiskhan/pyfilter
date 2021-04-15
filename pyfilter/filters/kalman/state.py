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

    def resample(self, inds):
        self.utf.mean.values[:] = choose(self.utf.mean.values, inds)
        self.utf.cov[:] = choose(self.utf.cov, inds)

        self.utf.ym[:] = choose(self.utf.ym, inds)
        self.utf.yc[:] = choose(self.utf.yc, inds)

        self.ll[:] = choose(self.ll, inds)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, inds):
        self.utf.mean.values[inds] = state.utf.mean.values[inds]
        self.utf.cov[inds] = state.utf.cov[inds]

        self.utf.ym[inds] = state.utf.ym[inds]
        self.utf.yc[inds] = state.utf.yc[inds]

        self.ll[inds] = state.ll[inds]
