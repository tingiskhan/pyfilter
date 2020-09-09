from ..uft import UFTCorrectionResult
from torch import Tensor
from ..utils import choose


class BaseState(object):
    def get_mean(self) -> Tensor:
        raise NotImplementedError()

    def resample(self, inds: Tensor):
        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        raise NotImplementedError()

    def exchange(self, inds: Tensor, state):
        raise NotImplementedError()


class KalmanState(BaseState):
    def __init__(self, utf: UFTCorrectionResult, ll: Tensor):
        self.utf = utf
        self.ll = ll

    def get_mean(self):
        return self.utf.xm

    def resample(self, inds):
        self.utf.xm = choose(self.utf.xm, inds)
        self.utf.xc = choose(self.utf.xc, inds)

        self.utf.ym = choose(self.utf.ym, inds)
        self.utf.yc = choose(self.utf.yc, inds)

    def get_loglikelihood(self):
        return self.ll


class ParticleState(BaseState):
    def __init__(self, x: Tensor, w: Tensor, ll: Tensor):
        self.x = x
        self.w = w
        self.ll = ll

    def get_mean(self):
        if self.x.dim() == self.w.dim() + 1:
            return (self.x * self.w.unsqueeze(-1)).sum(-2)
        elif self.x.dim() == self.w.dim():
            return (self.x * self.w).sum(-1)

        raise NotImplementedError()

    def resample(self, inds):
        self.x = choose(self.x, inds)
        self.w = choose(self.w, inds)

    def get_loglikelihood(self):
        return self.ll