from .pf import ParticleFilter
from ..utils import loglikelihood, choose
from ..normalization import normalize
import torch
from .state import ParticleState


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def _filter(self, y, state: ParticleState):
        # ===== Resample among old ===== #
        # TODO: Not optimal as we normalize in several other functions, fix this
        old_normw = normalize(state.w)

        inds, mask = self._resample_state(state.w)
        to_prop = choose(state.x, inds)
        self._proposal = self.proposal.construct(y, to_prop)

        # ===== Propagate ===== #
        x = self.proposal.draw(self._rsample)
        weights = self.proposal.weight(y, x, to_prop)

        self.proposal.resample(inds)

        # ===== Update weights ===== #
        tw = torch.zeros_like(weights)
        tw[~mask] = state.w[~mask]

        w = weights + tw

        return ParticleState(x, w, loglikelihood(weights, old_normw))