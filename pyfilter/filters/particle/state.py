from torch import Tensor
from ..state import BaseState
from ...utils import choose, normalize
from ...timeseries import NewState


class ParticleState(BaseState):
    def __init__(self, x: NewState, w: Tensor, ll: Tensor, prev_inds: Tensor):
        super().__init__()
        self.x = x
        self.register_buffer("w", w)
        self.register_buffer("ll", ll)
        self.register_buffer("prev_inds", prev_inds)

    def get_mean(self):
        normalized_weights = self.normalized_weights()
        if self.x.values.dim() == normalized_weights.dim() + 1:
            return (self.x.values * normalized_weights.unsqueeze(-1)).sum(-2)
        elif self.x.values.dim() == normalized_weights.dim():
            return (self.x.values * normalized_weights).sum(-1)

        raise NotImplementedError()

    def normalized_weights(self):
        return normalize(self.w)

    def resample(self, indices):
        self.x.values[:] = choose(self.x.values, indices)
        self.w[:] = choose(self.w, indices)
        self.ll[:] = choose(self.ll, indices)
        self.prev_inds[:] = choose(self.prev_inds, indices)

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, indices):
        self.x.values[indices] = state.x.values[indices]
        self.w[indices] = state.w[indices]
        self.ll[indices] = state.ll[indices]
        self.prev_inds[indices] = state.prev_inds[indices]
