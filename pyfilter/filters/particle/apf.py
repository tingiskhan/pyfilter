import torch

from ..utils import batched_gather
from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterState
from .utils import log_likelihood


class APF(ParticleFilter):
    """
    Implements the `Auxiliary Particle Filter`_ of Pitt and Shephard.

    .. _`Auxiliary Particle Filter`: https://en.wikipedia.org/wiki/Auxiliary_particle_filter
    """

    def predict(self, state: ParticleFilterState):
        normalized = state.normalized_weights()
        old_indices = torch.zeros_like(state.previous_indices) + torch.arange(normalized.shape[-1], device=normalized.device)

        return ParticleFilterPrediction(state.timeseries_state, normalized, old_indices)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction):
        pre_weights = self.proposal.pre_weight(y, state.timeseries_state)

        resample_weights = pre_weights + state.weights

        indices = self._resampler(resample_weights)

        dim = len(self.batch_shape)
        resampled_x = batched_gather(state.timeseries_state.value, indices, dim)
        resampled_state = state.timeseries_state.copy(values=resampled_x)

        x, weights = self._proposal.sample_and_weight(y, resampled_state)

        w = weights - pre_weights.gather(dim, indices)
        ll = log_likelihood(w) + (prediction.old_weights * pre_weights.exp()).sum(dim=-1).log()

        return ParticleFilterState(x, w, ll, indices)
