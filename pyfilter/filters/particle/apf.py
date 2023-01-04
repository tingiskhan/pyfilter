import torch

from ..utils import batched_gather
from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterCorrection
from .utils import log_likelihood


class APF(ParticleFilter):
    """
    Implements the `Auxiliary Particle Filter`_ of Pitt and Shephard.

    .. _`Auxiliary Particle Filter`: https://en.wikipedia.org/wiki/Auxiliary_particle_filter
    """

    def predict(self, state):
        normalized_weigths = state.normalized_weights()
        old_indices = torch.zeros_like(state.previous_indices) + torch.arange(normalized_weigths.shape[-1], device=normalized_weigths.device)

        return ParticleFilterPrediction(state.timeseries_state, state.weights, normalized_weigths, old_indices)

    def correct(self, y, prediction):
        timeseries_state = prediction.get_timeseries_state()
        pre_weights = self.proposal.pre_weight(y, timeseries_state)

        resample_weights = pre_weights + prediction.weights

        indices = self._resampler(resample_weights)

        dim = len(self.batch_shape)
        resampled_x = timeseries_state.copy(values=batched_gather(timeseries_state.value, indices, dim))
        resampled_prediction = ParticleFilterPrediction(resampled_x, prediction.weights, prediction.normalized_weights, prediction.indices)

        x, weights = self._proposal.sample_and_weight(y, resampled_prediction)

        w = weights - pre_weights.gather(dim, indices)
        ll = log_likelihood(w) + (prediction.normalized_weights * pre_weights.exp()).sum(dim=-1).log()

        return ParticleFilterCorrection(x, w, ll, indices)
