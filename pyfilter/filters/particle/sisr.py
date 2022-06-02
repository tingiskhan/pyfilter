from typing import Tuple

from .base import ParticleFilter
from .utils import log_likelihood
import torch
from .state import ParticleFilterState, ParticleFilterPrediction
from ..utils import batched_gather
from ...utils import get_ess


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def _resample_parallel(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        ess: torch.Tensor = get_ess(w) / w.shape[-1]
        mask: torch.BoolTensor = ess < self._resample_threshold

        return self._resampler(w[mask]), mask

    def predict(self, state):
        old_normalized_w = state.normalized_weights()

        indices, mask = self._resample_parallel(state.w)

        resampled_x = state.x.values
        resampled_x[mask] = batched_gather(resampled_x[mask], indices, len(self.batch_shape))

        resampled_state = state.x.copy(values=resampled_x)

        return ParticleFilterPrediction(resampled_state, old_normalized_w, indices=indices, mask=mask)

    def correct(self, y: torch.Tensor, state, prediction: ParticleFilterPrediction):
        x, weights = self.proposal.sample_and_weight(y, prediction.prev_x)

        tw = torch.zeros_like(weights)
        tw[~prediction.mask] = state.w[~prediction.mask]

        w = weights + tw

        return ParticleFilterState(x, w, log_likelihood(weights, prediction.old_weights), prediction.indices)
