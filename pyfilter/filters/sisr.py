from .base import ParticleFilter
from ..utils import loglikelihood, choose
from ..normalization import normalize
import torch


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def _filter(self, y):
        # ===== Resample among old ===== #
        # TODO: Not optimal as we normalize in several other functions, fix this
        old_normw = normalize(self._w_old)

        inds, mask = self._resample_state(self._w_old)
        to_prop = choose(self._x_cur, inds)
        self._proposal = self.proposal.construct(y, to_prop)

        # ===== Propagate ===== #
        self._x_cur = self.proposal.draw(self._rsample)
        weights = self.proposal.weight(y, self._x_cur, to_prop)

        self.proposal.resample(inds)

        # ===== Update weights ===== #
        tw = torch.zeros_like(weights)
        tw[~mask] = self._w_old[~mask]

        self._w_old = weights + tw

        normw = normalize(self._w_old)
        if self._sumaxis < -1:
            normw.unsqueeze_(-1)

        return (normw * self._x_cur).sum(self._sumaxis), loglikelihood(weights, old_normw)