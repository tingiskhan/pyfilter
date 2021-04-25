from .base import ParticleFilter
from ...utils import loglikelihood, choose
import torch
from .state import ParticleState


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def predict_correct(self, y, state: ParticleState) -> ParticleState:
        old_normalized_w = state.normalized_weights()

        inds, mask = self._resample_state(state.w)
        state.x.values[:] = choose(state.x.values, inds)

        if torch.isnan(y).any():
            x = self.ssm.hidden.forward(state.x)
            return ParticleState(x, 0.0 * old_normalized_w, 0.0 * state.get_loglikelihood(), inds)

        x, weights = self.proposal.sample_and_weight(y, state.x)

        tw = torch.zeros_like(weights)
        tw[~mask] = state.w[~mask]

        w = weights + tw

        return ParticleState(x, w, loglikelihood(weights, old_normalized_w), inds)
