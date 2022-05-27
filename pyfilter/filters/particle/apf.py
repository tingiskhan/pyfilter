import torch
from .base import ParticleFilter
from .utils import log_likelihood
from .state import ParticleFilterState, ParticleFilterPrediction


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """

    def predict(self, state: ParticleFilterState):
        normalized = state.normalized_weights()

        return ParticleFilterPrediction(state.x, normalized, None, None)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction):
        pre_weights = self.proposal.pre_weight(y, state.x)

        resample_weights = pre_weights + state.w

        indices = self._resampler(resample_weights)
        resampled_x = state.x.copy(values=state.x.values.index_select(dim=len(self.batch_shape), index=indices))

        x, weights = self._proposal.sample_and_weight(y, resampled_x)

        w = weights - pre_weights.index_select(dim=-1, index=indices)
        ll = log_likelihood(w) + (prediction.old_weights * pre_weights.exp()).sum(-1).log()

        return ParticleFilterState(x, w, ll, indices)
