import torch

from ..utils import batched_gather
from .base import ParticleFilter
from .state import ParticleFilterCorrection, ParticleFilterPrediction
from .utils import log_likelihood


class APF(ParticleFilter):
    """
    Implements the `Auxiliary Particle Filter`_ of Pitt and Shephard.

    .. _`Auxiliary Particle Filter`: https://en.wikipedia.org/wiki/Auxiliary_particle_filter
    """

    def predict(self, state):
        normalized_weigths = state.normalized_weights()
        old_indices = torch.arange(normalized_weigths.shape[0], device=normalized_weigths.device)

        if self.batch_shape:
            old_indices = old_indices.unsqueeze(-1).expand(self.particles)

        return ParticleFilterPrediction(state.timeseries_state, state.weights, normalized_weigths, old_indices)

    def correct(self, y, prediction):
        timeseries_state = prediction.get_timeseries_state()
        pre_weights = self.proposal.pre_weight(y, timeseries_state)

        resample_weights = pre_weights + prediction.weights

        indices = self._resampler(resample_weights)

        dim = 0
        resampled_x = timeseries_state.copy(values=batched_gather(timeseries_state.value, indices, dim))

        temp_weights = torch.zeros_like(resample_weights)
        resampled_prediction = ParticleFilterPrediction(
            resampled_x, temp_weights, temp_weights + 1.0 / pre_weights.shape[0], None
        )

        x, weights = self._proposal.sample_and_weight(y, resampled_prediction)

        weights = weights - pre_weights.gather(dim, indices)
        ll = log_likelihood(weights) + (prediction.normalized_weights * pre_weights.exp()).sum(dim=0).log()

        return ParticleFilterCorrection(x, weights, ll, indices)
