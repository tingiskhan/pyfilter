from torch import Tensor
from ..state import BaseState
from ...utils import choose, normalize
from ...timeseries import TimeseriesState


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
        self.x.state[:] = choose(self.x.state, inds)
        self.w[:] = choose(self.w, inds)
        self.ll[:] = choose(self.ll, inds)
        self.prev_inds[:] = choose(self.prev_inds, inds)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, inds):
        self.x.state[inds] = state.x.state[inds]
        self.w[inds] = state.w[inds]
        self.ll[inds] = state.ll[inds]
        self.prev_inds[inds] = state.prev_inds[inds]
