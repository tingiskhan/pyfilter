from typing import Tuple

from .base import ParticleFilter
from .utils import log_likelihood
import torch
from .state import ParticleFilterState, ParticleFilterPrediction
from ..utils import batched_gather
from ...utils import get_ess


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
        old_normalized_w = state.normalized_weights()
        indices, mask = self._resample_parallel(state.w)

        # TODO: Perhaps slow?
        all_indices = torch.empty_like(state.prev_inds)
        all_indices[mask] = indices
        all_indices[~mask] = torch.arange(0, state.prev_inds.shape[-1], device=all_indices.device)

        resampled_x = state.x.values
        resampled_x[mask] = batched_gather(resampled_x[mask], indices, indices.dim() - 1)

        resampled_state = state.x.copy(values=resampled_x)

        return ParticleFilterPrediction(resampled_state, old_normalized_w, indices=all_indices, mask=mask)

    def correct(self, y: torch.Tensor, state, prediction: ParticleFilterPrediction):
        x, weights = self.proposal.sample_and_weight(y, prediction.prev_x)

        tw = torch.zeros_like(weights)
        tw[~prediction.mask] = state.w[~prediction.mask]

        w = weights + tw

        return ParticleFilterState(x, w, log_likelihood(weights, prediction.old_weights), prediction.indices)
