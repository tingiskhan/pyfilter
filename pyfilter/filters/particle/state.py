from collections import OrderedDict
from typing import Any, Dict

import torch
from stochproc.timeseries import TimeseriesState, StateSpaceModel
from torch import Tensor
from pyro.distributions import Distribution, Normal, MultivariateNormal

from ...utils import normalize
from ..state import Correction, Prediction
from ..utils import batched_gather
from .utils import get_filter_mean_and_variance


class ParticleFilterPrediction(Prediction):
    """
    Prediction state for particle filters.
    """

    def __init__(self, prev_x: TimeseriesState, weights: Tensor, normalized_weights: Tensor, indices: Tensor):
        """
        Internal initializer for :class:`ParticleFilterPrediction`.

        Args:
            prev_x: resampled previous state.
            weigths: unnormalized weights.
            normalized_weights: normalized weights.
            indices: resampled indices.
        """

        self.prev_x = prev_x
        self.weights = weights
        self.normalized_weights = normalized_weights
        self.indices = indices

    def get_timeseries_state(self) -> TimeseriesState:
        return self.prev_x

    def create_state_from_prediction(self, model):
        x_new = model.hidden.propagate(self.get_timeseries_state())
        new_ll = torch.zeros(self.normalized_weights.shape[:-1], device=x_new.value.device)

        return ParticleFilterCorrection(x_new, self.weights, new_ll, self.indices)

    def get_predictive_density(self, model: StateSpaceModel, approximate: bool = False) -> Distribution:
        """
        Constructs the approximation of the predictive distribution.

        Args:
            model (StateSpaceModel): model to use for constructing predictive density.
            approximate (bool): whether to approximate the predictive distribution via Gaussians.

        Returns:
            Distribution: predictive distribution distribution.
        """

        if not approximate:
            return model.hidden.build_density(self.get_timeseries_state())

        x_new = model.hidden.propagate(self.get_timeseries_state())
        keep_dim = (self.weights.dim() > 1) and (self.prev_x.event_shape.numel() > 1)
        mean, var = get_filter_mean_and_variance(x_new, self.normalized_weights, covariance=True, keep_dim=keep_dim)

        if model.hidden.n_dim == 0:
            return Normal(mean, var.sqrt())
        
        return MultivariateNormal(mean, covariance_matrix=var)


class ParticleFilterCorrection(Correction):
    """
    State object for particle based filters.
    """

    def __init__(self, x: TimeseriesState, w: Tensor, ll: Tensor, prev_indices: Tensor):
        """
        Internal initializer for :class:`ParticleFilterState`.

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

        self["_mean"], self["_var"] = get_filter_mean_and_variance(x, self.normalized_weights())

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

        # TODO: Resample instead...?
        self["_mean"], self["_var"] = get_filter_mean_and_variance(self["_x"], self.normalized_weights())

    def exchange(self, state: "ParticleFilterCorrection", mask):
        self["_x"].value[mask] = state.timeseries_state.value[mask]
        self["_w"][mask] = state.weights[mask]
        self["_ll"][mask] = state.get_loglikelihood()[mask]
        self["_prev_inds"][mask] = state.previous_indices[mask]

        # TODO: Resample instead...?
        self["_mean"], self["_var"] = get_filter_mean_and_variance(self["_x"], self.normalized_weights())

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
