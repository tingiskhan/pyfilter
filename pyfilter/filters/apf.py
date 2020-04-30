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
        self.proposal.construct(y, self._x_cur)
        pre_weights = self.ssm.log_prob(y, self.ssm.hidden.prop_apf(self._x_cur))

        resamp_w = pre_weights + self._w_old
        normalized = normalize(self._w_old)

        # ===== Resample and propagate ===== #
        resampled_indices = self._resampler(resamp_w)
        resampled_x = choose(self._x_cur, resampled_indices)

        self._proposal = self.proposal.resample(resampled_indices)
        self._x_cur = self._proposal.draw(self._rsample)

        weights = self.proposal.weight(y, self._x_cur, resampled_x)

        self._w_old = weights - choose(pre_weights, resampled_indices)

        # ===== Calculate log likelihood ===== #
        ll = loglikelihood(self._w_old) + torch.log((normalized * torch.exp(pre_weights)).sum(-1))

        # ===== Get weights ====== #
        normw = normalize(self._w_old)
        if self._sumaxis < -1:
            normw.unsqueeze_(-1)

        return (normw * self._x_cur).sum(self._sumaxis), ll