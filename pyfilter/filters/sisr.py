from .base import ParticleFilter
from ..utils.utils import loglikelihood, choose
from ..utils.normalization import normalize


class SISR(ParticleFilter):
    """
    Implements the SISR filter by Gordon et al.
    """

    def _filter(self, y):
        # ===== Resample among old ===== #
        inds = self._resamp(self._w_old)
        to_prop = choose(self._x_cur, inds)
        self._proposal = self._proposal.resample(inds)

        # ===== Propagate ===== #
        self._x_cur = self._proposal.draw(y, to_prop, size=self._particles)
        weights = self._proposal.weight(y, self._x_cur, to_prop)

        self._w_old = weights

        return (normalize(weights) * self._x_cur).sum(-1), loglikelihood(weights)