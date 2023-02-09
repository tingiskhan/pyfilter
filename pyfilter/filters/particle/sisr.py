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
        mask = ess < self._resample_threshold
        
        ts_state = state.get_timeseries_state()
        weights = state.weights
        prev_inds = state.previous_indices

        if not mask.any():
            return ParticleFilterPrediction(ts_state, weights, normalized_weigths, indices=prev_inds)

        # Resample
        sub_indices = self._resampler(normalized_weigths[..., mask], normalized=True)
        
        unsqueezed_mask = mask.unsqueeze(0)
        resampled_indices = prev_inds.masked_scatter(unsqueezed_mask, sub_indices)
        
        resampled_weights = weights.masked_fill(unsqueezed_mask, 0.0)
        normalized_weigths = normalized_weigths.masked_fill(unsqueezed_mask, 1.0 / weights.shape[0])
        
        ts_vals = ts_state.value
        temp_resampled = ts_vals[sub_indices, mask]

        if self._model.n_dim > 0:
            unsqueezed_mask.unsqueeze_(-1)

        resampled_x = ts_vals.masked_scatter(unsqueezed_mask, temp_resampled)
        resampled_state = ts_state.copy(values=resampled_x)
                            
        return ParticleFilterPrediction(resampled_state, resampled_weights, normalized_weigths, indices=resampled_indices)

    def correct(self, y, prediction):
        x, weights = self.proposal.sample_and_weight(y, prediction)
        new_weights = weights + prediction.weights

        return ParticleFilterCorrection(
            x, new_weights, log_likelihood(weights, prediction.normalized_weights), prediction.indices
        )
