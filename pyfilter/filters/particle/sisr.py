from typing import Tuple

import torch

from ...utils import get_ess
from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterCorrection
from .utils import log_likelihood


class SISR(ParticleFilter):
    """
    Implements the `Sequential Importance Sampling Resampling`_ filter by Gordon et al.

    .. _`Sequential Importance Sampling Resampling`: https://en.wikipedia.org/wiki/Particle_filter#Sequential_Importance_Resampling_(SIR)
    """

    def _resample_parallel(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        ess: torch.Tensor = get_ess(w) / w.shape[0]
        mask: torch.BoolTensor = ess < self._resample_threshold

        return self._resampler(w[..., mask]), mask

    def predict(self, state):
        normalized_weigths = state.normalized_weights()
        resampled_indices, mask = self._resample_parallel(state.weights)

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

        return ParticleFilterCorrection(x, new_weights, log_likelihood(weights, prediction.normalized_weights), prediction.indices)
