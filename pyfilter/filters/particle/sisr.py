from .base import ParticleFilter
from .utils import log_likelihood
from ...utils import choose
import torch
from .state import ParticleFilterState, ParticleFilterPrediction


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def predict(self, state):
        old_normalized_w = state.normalized_weights()

        indices, mask = self._resample_parallel(state.w)
        resampled_x = state.x.copy(values=choose(state.x.values, indices))

        new_x = self._model.hidden.propagate(resampled_x)

        return ParticleFilterPrediction(new_x, old_normalized_w, indices=indices, mask=mask)

    def correct(self, y: torch.Tensor, state, prediction: ParticleFilterPrediction):
        x, weights = self.proposal.sample_and_weight(y, prediction.x)

        tw = torch.zeros_like(weights)
        tw[~prediction.mask] = state.w[~prediction.mask]

        w = weights + tw

        return ParticleFilterState(x, w, log_likelihood(weights, prediction.old_weights), prediction.indices)
