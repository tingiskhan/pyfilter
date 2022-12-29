from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
from stochproc.timeseries import TimeseriesState
from torch import Tensor

from ...utils import normalize
from ..state import FilterState, PredictionState
from ..utils import batched_gather


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
        new_ll = torch.zeros(self.old_weights.shape[:-1], device=x_new.value.device)

        # TODO: Indicies is wrong
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
        self["_x"] = x
        self["_w"] = w
        self["_ll"] = ll
        self["_prev_inds"] = prev_indices

        self["_mean"], self["_var"] = self._calc_mean_and_var()

    @property
    def timeseries_state(self) -> TimeseriesState:
        return self["_x"]

    @property
    def weights(self) -> torch.Tensor:
        return self["_w"]
    
    @property
    def previous_indices(self) -> torch.Tensor:
        return self["_prev_inds"]
    
    def get_loglikelihood(self):
        return self["_ll"]

    def get_mean(self) -> Tensor:
        return self["_mean"]
    
    def get_variance(self) -> Tensor:
        return self["_var"]

    def _calc_mean_and_var(self) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_weights = self.normalized_weights()

        sum_axis = -(len(self.timeseries_state.event_shape) + 1)
        nested = self.weights.dim() > 1

        if sum_axis < -1:
            normalized_weights.unsqueeze_(-1)

        mean = (self.timeseries_state.value * normalized_weights).sum(sum_axis)
        var = ((self.timeseries_state.value - (mean if not nested else mean.unsqueeze(1))) ** 2 * normalized_weights).sum(sum_axis)

        return mean, var

    # TODO: Covariance doesn't work for nested
    def get_covariance(self) -> torch.Tensor:
        """
        Returns the covariance of the posterior distribution.
        """

        if len(self.timeseries_state.event_shape) == 0:
            return self.get_variance()
        
        # TODO: Duplication
        w = self.normalized_weights()
        x = self.timeseries_state.value

        mean = w.unsqueeze(-2) @ x
        centralized = x - mean
        weighted_covariance = w.view(w.shape + torch.Size([1, 1])) * (centralized.unsqueeze(-1) @ centralized.unsqueeze(-2))
        
        return weighted_covariance.sum(dim=-3)

    def normalized_weights(self):
        return normalize(self.weights)

    def resample(self, indices):
        resampled_values = batched_gather(
            self.timeseries_state.value,
            indices,
            self.timeseries_state.value.dim() - self.timeseries_state.event_shape.numel()
        )
                
        self["_x"] = self.timeseries_state.copy(values=resampled_values)
        self["_w"] = batched_gather(self.weights, indices, 0)
        self["_ll"] = batched_gather(self.get_loglikelihood(), indices, 0)
        self["_prev_inds"] = batched_gather(self.previous_indices, indices, 1)

        self["_mean"], self["_var"] = self._calc_mean_and_var()

    def exchange(self, state: "ParticleFilterState", mask):
        self["_x"].value[mask] = state.timeseries_state.value[mask]
        self["_w"][mask] = state.weights[mask]
        self["_ll"][mask] = state.get_loglikelihood()[mask]
        self["_prev_inds"][mask] = state.previous_indices[mask]

        self["_mean"], self["_var"] = self._calc_mean_and_var()

    def get_timeseries_state(self) -> TimeseriesState:
        return self.timeseries_state

    def predict_path(self, model, num_steps):
        return model.sample_states(num_steps, x_0=self.timeseries_state)

    def state_dict(self) -> Dict[str, Any]:
        result = OrderedDict([])

        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                result[k] = v
        
        result["_x"] = {
            "time_index": self.timeseries_state.time_index,
            "value": self.timeseries_state.value
        }

        return result

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads state from existing state dictionary.

        Args:
            state_dict: state dictionary to load from.
        """

        # TODO: Handle case when the particles have doubled better?
        values_to_load = state_dict["_x"]["value"]
        msg = f"Seems like you're loading a different shape: self:{self.timeseries_state.value.shape} != other:{values_to_load.shape}"
        assert self.timeseries_state.value.shape == values_to_load.shape, msg

        self["_x"] = self.timeseries_state.propagate_from(
            values=values_to_load, 
            time_increment=-self.timeseries_state.time_index + state_dict["_x"]["time_index"]
        )

        self["_w"] = state_dict["_w"]
        self["_ll"] = state_dict["_ll"]
        self["_prev_inds"] = state_dict["_prev_inds"]

        self["_mean"], self["_var"] = state_dict["_mean"], state_dict["_var"]

    def __repr__(self):
        return f"{self.__class__.__name__}(time_index: {self.timeseries_state.time_index}, event_shape: {self.timeseries_state.event_shape})"
