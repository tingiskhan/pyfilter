import torch
from torch import Tensor
from stochproc.timeseries import TimeseriesState
from ..state import FilterState, PredictionState
from ...utils import choose, normalize


class ParticleFilterPrediction(PredictionState):
    """
    Prediction state for particle filters.
    """

    def __init__(self, prev_x: TimeseriesState, old_weights: Tensor, indices: Tensor, mask: Tensor = None):
        """
        Initializes the :class:`ParticleFilterPrediction` class.

        Args:
            prev_x: the resampled previous state.
            old_weights: the previous normalized weights.
            indices: the selected mask
            mask: mask for which batch to resample, only relevant for filters in parallel.
        """

        self.prev_x = prev_x
        self.old_weights = old_weights
        self.indices = indices
        self.mask = mask


class ParticleFilterState(FilterState):
    """
    State object for particle based filters.
    """

    def __init__(self, x: TimeseriesState, w: Tensor, ll: Tensor, prev_indices: Tensor):
        """
        Initializes the :class:`ParticleFilterState` class.

        Args:
            x: the state particles of the timeseries.
            w: the log weights associated with the state particles.
            ll: the estimate log-likelihood, i.e. :math:`p(y_t)`.
            prev_indices: the mask of the previous state particles that were used to propagate to this state.
        """

        super().__init__()
        self.x = x
        self.w = w
        self.ll = ll
        self.prev_inds = prev_indices

        mean, var = self._calc_mean_and_var()
        self.mean = mean
        self.var = var

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
        return self.mean

    def get_variance(self):
        return self.var

    def normalized_weights(self):
        return normalize(self.w)

    # TODO: Use batched_gather instead
    def resample(self, indices):
        self.__init__(
            self.x.copy(values=choose(self.x.values, indices)),
            choose(self.w, indices),
            choose(self.ll, indices),
            choose(self.prev_inds, indices),
        )

    def get_loglikelihood(self):
        return self.ll

    # TODO: Use batched_gather instead
    # TODO: Improve...
    def exchange(self, state: "ParticleFilterState", mask):
        x = self.x.copy(values=self.x.values.clone())
        x.values[mask] = state.x.values[mask]

        w = self.w.clone()
        w[mask] = state.w[mask]

        ll = self.ll.clone()
        ll[mask] = state.ll[mask]

        prev_inds = self.prev_inds.clone()
        prev_inds[mask] = state.prev_inds[mask]

        self.__init__(x, w, ll, prev_inds)

    def get_timeseries_state(self) -> TimeseriesState:
        return self.x
