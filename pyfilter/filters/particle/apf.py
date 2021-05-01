import torch
from .base import ParticleFilter
from ...utils import loglikelihood, choose
from .state import ParticleState


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """

    def predict_correct(self, y, state: ParticleState):
        normalized = state.normalized_weights()

        if torch.isnan(y).any():
            indices = self._resampler(normalized, normalized=True)
            state.x.values[:] = choose(state.x.values, indices)

            x = self.ssm.hidden.forward(state.x)

            return ParticleState(x, torch.zeros_like(normalized), 0.0 * state.get_loglikelihood(), indices)

        pre_weights = self.proposal.pre_weight(y, state.x)

        resample_weights = pre_weights + state.w

        resampled_indices = self._resampler(resample_weights)
        state.x.values[:] = choose(state.x.values, resampled_indices)

        x, weights = self._proposal.sample_and_weight(y, state.x)

        w = weights - choose(pre_weights, resampled_indices)
        ll = loglikelihood(w) + torch.log((normalized * torch.exp(pre_weights)).sum(-1))

        return ParticleState(x, w, ll, resampled_indices)
