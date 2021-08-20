import torch
from .base import ParticleFilter
from ...utils import loglikelihood, choose
from .state import ParticleFilterState


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """

    def predict_correct(self, y, state: ParticleFilterState):
        normalized = state.normalized_weights()

        if torch.isnan(y).any():
            indices = self._resampler(normalized, normalized=True)
            copied_x = state.x.copy(values=choose(state.x.values, indices))

            x = self.ssm.hidden.forward(copied_x)

            return ParticleFilterState(x, torch.zeros_like(normalized), 0.0 * state.get_loglikelihood(), indices)

        pre_weights = self.proposal.pre_weight(y, state.x)

        resample_weights = pre_weights + state.w

        indices = self._resampler(resample_weights)
        copied_x = state.x.copy(values=choose(state.x.values, indices))

        x, weights = self._proposal.sample_and_weight(y, copied_x)

        w = weights - choose(pre_weights, indices)
        ll = loglikelihood(w) + torch.log((normalized * pre_weights.exp()).sum(-1))

        return ParticleFilterState(x, w, ll, indices)
