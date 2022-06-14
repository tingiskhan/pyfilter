import torch
from torch import Tensor
from stochproc.timeseries import TimeseriesState
from ..state import FilterState, PredictionState
from ..utils import batched_gather
from ...utils import normalize


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

    def get_previous_state(self) -> TimeseriesState:
        return self.prev_x

    def create_state_from_prediction(self, model):
        x_new = model.hidden.propagate(self.prev_x)
        new_ll = torch.zeros(self.old_weights.shape[:-1], device=x_new.values.device)

        return ParticleFilterState(x_new, torch.zeros_like(self.old_weights), new_ll, self.indices)


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

        sum_axis = -(len(self.x.event_dim) + 1)
        nested = self.w.dim() > 1

        if sum_axis < -1:
            normalized_weights.unsqueeze_(-1)

        mean = (self.x.values * normalized_weights).sum(sum_axis)
        var = ((self.x.values - (mean if not nested else mean.unsqueeze(1))) ** 2 * normalized_weights).sum(sum_axis)

        return mean, var

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.var

    def normalized_weights(self):
        return normalize(self.w)

    def resample(self, indices):
        self.__init__(
            self.x.copy(values=batched_gather(self.x.values, indices, self.x.values.dim() - self.x.event_dim.numel())),
            batched_gather(self.w, indices, 0),
            batched_gather(self.ll, indices, 0),
            batched_gather(self.prev_inds, indices, 1),
        )

    def get_loglikelihood(self):
        return self.ll

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

    def predict_path(self, model, num_steps):
        return model.sample_states(num_steps, x_0=self.x)
