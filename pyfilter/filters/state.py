from torch import Tensor
from torch.nn import Module
from ..uft import UFTCorrectionResult
from ..utils import choose, normalize
from ..timeseries import TimeseriesState


class BaseState(Module):
    def get_mean(self) -> Tensor:
        raise NotImplementedError()

    def resample(self, inds: Tensor):
        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        raise NotImplementedError()

    def exchange(self, state, inds: Tensor):
        raise NotImplementedError()


class KalmanState(BaseState):
    def __init__(self, utf: UFTCorrectionResult, ll: Tensor):
        super().__init__()
        self.add_module("utf", utf)
        self.register_buffer("ll", ll)

    def get_mean(self):
        return self.utf.xm

    def resample(self, inds):
        self.utf.mean = choose(self.utf.mean, inds)
        self.utf.cov = choose(self.utf.cov, inds)

        self.utf.ym = choose(self.utf.ym, inds)
        self.utf.yc = choose(self.utf.yc, inds)

        self.ll = choose(self.ll, inds)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, inds):
        self.utf.mean[inds] = state.utf.mean[inds]
        self.utf.cov[inds] = state.utf.cov[inds]

        self.utf.ym[inds] = state.utf.ym[inds]
        self.utf.yc[inds] = state.utf.yc[inds]

        self.ll[inds] = state.ll[inds]


class ParticleState(BaseState):
    def __init__(self, x: TimeseriesState, w: Tensor, ll: Tensor, prev_inds: Tensor):
        super().__init__()
        self.add_module("x", x)
        self.register_buffer("w", w)
        self.register_buffer("ll", ll)
        self.register_buffer("prev_inds", prev_inds)

    def get_mean(self):
        normw = self.normalized_weights()
        if self.x.state.dim() == self.w.dim() + 1:
            return (self.x.state * normw.unsqueeze(-1)).sum(-2)
        elif self.x.state.dim() == self.w.dim():
            return (self.x.state * normw).sum(-1)

        raise NotImplementedError()

    def normalized_weights(self):
        return normalize(self.w)

    def resample(self, inds):
        self.x.state = choose(self.x.state, inds)
        self.w = choose(self.w, inds)
        self.ll = choose(self.ll, inds)
        self.prev_inds = choose(self.prev_inds, inds)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, inds):
        self.x.state[inds] = state.x.state[inds]
        self.w[inds] = state.w[inds]
        self.ll[inds] = state.ll[inds]
        self.prev_inds[inds] = state.prev_inds[inds]
