from .base import ParticleFilter
from ..utils import loglikelihood, choose
from ..normalization import normalize
import torch


class APF(ParticleFilter):
    """
    Implements the Auxiliary Particle Filter of Pitt and Shephard.
    """
    def _filter(self, y):
        # ===== Perform auxiliary sampling ===== #
        t_x = self._model.propagate_apf(self._x_cur)
        t_weights = self._model.weight(y, t_x)

        resamp_w = t_weights + self._w_old
        normalized = normalize(self._w_old)

        # ===== Resample and propagate ===== #
        resampled_indices = self._resampler(resamp_w)
        resampled_x = choose(self._x_cur, resampled_indices)

        self._proposal = self._proposal.resample(resampled_indices)

        self._x_cur = self._proposal.draw(y, resampled_x)
        weights = self._proposal.weight(y, self._x_cur, resampled_x)

        self._w_old = weights - choose(t_weights, resampled_indices)

        # ===== Calculate log likelihood ===== #
        ll = loglikelihood(self._w_old) + torch.log((normalized * torch.exp(t_weights)).sum(-1))
        normw = normalize(self._w_old) if weights.dim() == self._x_cur.dim() else normalize(self._w_old).unsqueeze(-1)

        return (normw * self._x_cur).sum(self._sumaxis), ll