from typing import Tuple

import torch

from ...utils import get_ess
from .base import ParticleFilter
from .state import ParticleFilterCorrection, ParticleFilterPrediction
from .utils import log_likelihood


class SISR(ParticleFilter):
    """
    Implements the `Sequential Importance Sampling Resampling`_ filter by Gordon et al.

    .. _`Sequential Importance Sampling Resampling`: https://en.wikipedia.org/wiki/Particle_filter#Sequential_Importance_Resampling_(SIR)
    """    

    def predict(self, state):
        # Get indices for resampling
        normalized_weigths = state.normalized_weights()

        ess = get_ess(normalized_weigths, normalized=True)
        mask = ess < (self._resample_threshold * normalized_weigths.shape[0])

        resampled_indices = self._resampler(normalized_weigths[..., mask], normalized=True)

        # Resample
        all_indices = state.previous_indices.clone()
        all_indices.masked_scatter_(mask.unsqueeze(0), resampled_indices)

        resampled_x = state.timeseries_state.value.clone()
        resampled_x[:, mask] = resampled_x[resampled_indices, mask]
        resampled_state = state.timeseries_state.copy(values=resampled_x)

        unsqueezed_mask = mask.unsqueeze(0)
        weights = state.weights.masked_fill(unsqueezed_mask, 0.0)
        normalized_weigths = normalized_weigths.masked_fill(unsqueezed_mask, 1.0 / weights.shape[0])

        return ParticleFilterPrediction(resampled_state, weights, normalized_weigths, indices=all_indices)

    def correct(self, y, prediction):
        x, weights = self.proposal.sample_and_weight(y, prediction)
        new_weights = weights + prediction.weights

        return ParticleFilterCorrection(
            x, new_weights, log_likelihood(weights, prediction.normalized_weights), prediction.indices
        )
