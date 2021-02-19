import torch
from .base import ParticleFilter
from ...utils import loglikelihood, choose
from .state import ParticleState


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """

    def _filter(self, y, state: ParticleState):
        pre_weights = self.proposal.pre_weight(y, state.x)

        resamp_w = pre_weights + state.w
        normalized = state.normalized_weights()

        resampled_indices = self._resampler(resamp_w)
        state.x.state[:] = choose(state.x.state, resampled_indices)

        x, weights = self._proposal.sample_and_weight(y, state.x)

        w = weights - choose(pre_weights, resampled_indices)
        ll = loglikelihood(w) + torch.log((normalized * torch.exp(pre_weights)).sum(-1))

        return ParticleState(x, w, ll, resampled_indices)
