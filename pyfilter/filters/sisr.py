from .base import ParticleFilter
from ..utils.utils import loglikelihood, choose
from ..utils.normalization import normalize
import torch


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def _filter(self, y):
        # ===== Resample among old ===== #
        inds, mask = self._resample_state(self._w_old)
        to_prop = choose(self._x_cur, inds)
        self._proposal = self._proposal.resample(inds)

        # ===== Propagate ===== #
        self._x_cur = self._proposal.draw(y, to_prop, size=self._particles)
        weights = self._proposal.weight(y, self._x_cur, to_prop)

        # ===== Update weights ===== #
        tw = torch.zeros(weights.shape)
        tw[~mask] = self._w_old[~mask]

        self._w_old = weights + tw

        normw = normalize(weights) if weights.dim() == self._x_cur.dim() else normalize(weights).unsqueeze(-1)

        return (normw * self._x_cur).sum(self._sumaxis), loglikelihood(weights)