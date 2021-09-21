import torch
from torch import Tensor
from ..state import BaseFilterState
from ...utils import choose, normalize
from ...timeseries import NewState


class ParticleFilterState(BaseFilterState):
    """
    State object for particle based filters.
    """

    def __init__(self, x: NewState, w: Tensor, ll: Tensor, prev_indices: Tensor):
        """
        Initializes ``ParticleFilterState``.

        Args:
            x: The state particles of the timeseries.
            w: The log weights associated with the state particles.
            ll: The estimate log-likelihood, i.e. :math:`p(y_t)`.
            prev_indices: The indices of the previous state particles that were used to propagate to this state.
        """

        super().__init__()
        self.x = x
        self.register_buffer("w", w)
        self.register_buffer("ll", ll)
        self.register_buffer("prev_inds", prev_indices)

        mean, var = self._calc_mean_and_var()
        self.register_buffer("_mean", mean)
        self.register_buffer("_var", var)

    def _calc_mean_and_var(self) -> (torch.Tensor, torch.Tensor):
        normalized_weights = self.normalized_weights()

        sum_axis = -1
        if self.x.values.dim() == normalized_weights.dim() + 1:
            normalized_weights.unsqueeze_(-1)
            sum_axis = -2

        nested = self.w.dim() > 1

        mean = (self.x.values * normalized_weights).sum(sum_axis)
        var = ((self.x.values - (mean if not nested else mean.unsqueeze(1))) ** 2 * normalized_weights).sum(sum_axis)

        return mean, var

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._var

    def normalized_weights(self):
        return normalize(self.w)

    def resample(self, indices):
        self.__init__(
            self.x.copy(values=choose(self.x.values, indices)),
            choose(self.w, indices),
            choose(self.ll, indices),
            choose(self.prev_inds, indices),
        )

    def get_loglikelihood(self):
        return self.ll

    def exchange(self, state, indices):
        x = self.x.copy(values=self.x.values.clone())
        x.values[indices] = state.x.values[indices]

        w = self.w.clone()
        w[indices] = state.w[indices]

        ll = self.ll.clone()
        ll[indices] = state.ll[indices]

        prev_inds = self.prev_inds.clone()
        prev_inds[indices] = state.prev_inds[indices]

        self.__init__(x, w, ll, prev_inds)

    def get_timeseries_state(self) -> NewState:
        return self.x
