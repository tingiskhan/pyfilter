from typing import Tuple

import torch

from ...utils import get_ess
from ..utils import batched_gather
import torch

from ...utils import get_ess
from ..utils import batched_gather
from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterCorrection
from .utils import log_likelihood


class SISR(ParticleFilter):
    """
    Implements the `Sequential Importance Sampling Resampling`_ filter by Gordon et al.

    .. _`Sequential Importance Sampling Resampling`: https://en.wikipedia.org/wiki/Particle_filter#Sequential_Importance_Resampling_(SIR)
    """

    def _resample_parallel(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        ess: torch.Tensor = get_ess(w) / w.shape[-1]
        mask: torch.BoolTensor = ess < self._resample_threshold

        return self._resampler(w[mask]), mask

    def predict(self, state):
        normalized_weigths = state.normalized_weights()
        indices, mask = self._resample_parallel(state.weights)

        # TODO: Perhaps slow?
        all_indices = torch.empty_like(state.previous_indices)
        all_indices[mask] = indices
        all_indices[~mask] = torch.arange(0, state.previous_indices.shape[-1], device=all_indices.device)

        resampled_x = state.timeseries_state.value
        resampled_x[mask] = batched_gather(resampled_x[mask], indices, indices.dim() - 1)
        resampled_state = state.timeseries_state.copy(values=resampled_x)

        unsqueezed_mask = mask.unsqueeze(-1)
        weights = state.weights.masked_fill(unsqueezed_mask, 0.0)
        normalized_weigths = normalized_weigths.masked_fill(unsqueezed_mask, 1.0 / weights.shape[-1])

        return ParticleFilterPrediction(resampled_state, weights, normalized_weigths, indices=all_indices)

    # TODO: something wrong for SISR and linearized...
    def correct(self, y, prediction):
        x, weights = self.proposal.sample_and_weight(y, prediction)
        new_weights = weights + prediction.weights

        return ParticleFilterCorrection(x, new_weights, log_likelihood(weights, prediction.normalized_weights), prediction.indices)
