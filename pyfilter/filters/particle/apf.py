import torch
from .base import ParticleFilter
from .utils import log_likelihood
from .state import ParticleFilterState, ParticleFilterPrediction
from ..utils import batched_gather


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """

    def predict(self, state: ParticleFilterState):
        normalized = state.normalized_weights()
        old_indices = torch.zeros_like(state.prev_inds) + torch.arange(normalized.shape[-1], device=normalized.device)

        return ParticleFilterPrediction(state.x, normalized, old_indices)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction):
        pre_weights = self.proposal.pre_weight(y, state.x)

        resample_weights = pre_weights + state.w

        indices = self._resampler(resample_weights)

        dim = len(self.batch_shape)
        resampled_x = batched_gather(state.x.values, indices, dim)
        resampled_state = state.x.copy(values=resampled_x)

        x, weights = self._proposal.sample_and_weight(y, resampled_state)

        w = weights - pre_weights.gather(dim, indices)
        ll = log_likelihood(w) + (prediction.old_weights * pre_weights.exp()).sum(dim=-1).log()

        return ParticleFilterState(x, w, ll, indices)
