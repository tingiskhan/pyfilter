from .base import ParticleFilter
from ...utils import loglikelihood, choose
import torch
from .state import ParticleFilterState


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def predict_correct(self, y, state: ParticleFilterState) -> ParticleFilterState:
        old_normalized_w = state.normalized_weights()

        indices, mask = self._resample_state(state.w)
        copied_x = state.x.copy(values=choose(state.x.values, indices))

        if torch.isnan(y).any():
            x = self.ssm.hidden.forward(copied_x)
            return ParticleFilterState(x, 0.0 * old_normalized_w, 0.0 * state.get_loglikelihood(), indices)

        x, weights = self.proposal.sample_and_weight(y, copied_x)

        tw = torch.zeros_like(weights)
        tw[~mask] = state.w[~mask]

        w = weights + tw

        return ParticleFilterState(x, w, loglikelihood(weights, old_normalized_w), indices)
