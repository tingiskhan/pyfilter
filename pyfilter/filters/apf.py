from .pf import ParticleFilter
from ..utils import loglikelihood, choose
from ..normalization import normalize
import torch
from .state import ParticleState


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """

    def _filter(self, y, state: ParticleState):
        # ===== Perform auxiliary sampling ===== #
        pre_weights = self.proposal.pre_weight(y, state.x)

        resamp_w = pre_weights + state.w
        normalized = normalize(state.w)

        # ===== Resample ===== #
        resampled_indices = self._resampler(resamp_w)
        resampled_x = choose(state.x, resampled_indices)

        self.proposal.resample(resampled_indices)

        # ===== Construct ===== #
        self.proposal.construct(y, resampled_x)

        # ===== Propagate and weight ===== #
        x = self._proposal.draw(self._rsample)
        weights = self.proposal.weight(y, x, resampled_x)

        w = weights - choose(pre_weights, resampled_indices)

        # ===== Calculate log likelihood ===== #
        ll = loglikelihood(w) + torch.log((normalized * torch.exp(pre_weights)).sum(-1))

        return ParticleState(x, w, ll)
