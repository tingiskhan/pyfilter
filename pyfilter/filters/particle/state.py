from collections import OrderedDict
from typing import Any, Dict

import torch
from stochproc.timeseries import TimeseriesState, StateSpaceModel
from torch import Tensor
from pyro.distributions import Distribution, Normal, MultivariateNormal

from ...utils import normalize
from ..state import Correction, Prediction
from .utils import get_filter_mean_and_variance


class ParticleFilterPrediction(Prediction):
    """
    Prediction state for particle filters.
    """

    def __init__(self, prev_x: TimeseriesState, weights: Tensor, normalized_weights: Tensor, indices: Tensor):
        """
        Internal initializer for :class:`ParticleFilterPrediction`.

        Args:
            prev_x (TimeseriesState): resampled previous state.
            weights (Tensor): log-weights.
            normalized_weights (Tensor): normalized weights.
            indices (Tensor): resampled indices.
        """

        self.prev_x = prev_x
        self.weights = weights
        self.normalized_weights = normalized_weights
        self.indices = indices

    def get_timeseries_state(self) -> TimeseriesState:
        return self.prev_x

    def create_state_from_prediction(self, model):
        x_new = model.hidden.propagate(self.get_timeseries_state())
        new_ll = torch.zeros(self.normalized_weights.shape[1:], device=x_new.value.device)

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
        mean, var = get_filter_mean_and_variance(x_new, self.normalized_weights, covariance=True, keep_dim=False)

        if self.weights.dim() > 1:
            mean.unsqueeze_(0)
            var.unsqueeze_(0)
        
        if model.hidden.n_dim == 0:
            return Normal(mean, var.sqrt())
        
        return MultivariateNormal(mean, covariance_matrix=var)


class ParticleFilterCorrection(Correction):
    """
    State object for particle based filters.
    """

    def __init__(self, x: TimeseriesState, w: Tensor, ll: Tensor, prev_indices: Tensor):
        """
        Internal initializer for :class:`ParticleFilterCorrection`.

        Args:
            x (TimeseriesState): state particles of the timeseries.
            w (Tensor): log weights associated with the particles.
            ll (Tensor): estimated log-likelihood, i.e. :math:`p(y_t)`.
            prev_indices (Tensor): resampled indicies of particles.
        """

        super().__init__()
        self["_x"] = x
        self["_w"] = w
        self["_ll"] = ll
        self["_prev_inds"] = prev_indices

        # TODO: Speed up and use resampling instead
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

        # TODO: Fix this
        if len(self.timeseries_state.event_shape) == 0:
            return self.get_variance()
        
        # TODO: Duplication
        w = self.normalized_weights()
        x = self.timeseries_state.value

        mean = (w * x).sum(dim=0)
        centralized = x - mean
        weighted_covariance = w.view(w.shape + torch.Size([1, 1])) * (centralized.unsqueeze(-1) @ centralized.unsqueeze(-2))
        
        return weighted_covariance.sum(dim=0)

    # TODO: Cache?
    def normalized_weights(self) -> torch.Tensor:
        """
        Returns the normalized weights.

        Returns:
            torch.Tensor: normalized weights.
        """

        return normalize(self.weights)

    def resample(self, indices):
        self["_x"] = self.timeseries_state.copy(values=self.timeseries_state.value[:, indices])
        self["_w"] = self.weights[:, indices]

        self["_ll"][indices] = self["_ll"][indices]
        self["_prev_inds"] = self["_prev_inds"][:, indices]

        self["_mean"] = self["_mean"][indices]
        self["_var"] = self["_var"][indices]

    def exchange(self, other, mask):
        # TODO: Try to find a more general way of doing this...?
        self["_x"].value[:, mask] = other.timeseries_state.value[:, mask]
        self["_w"][:, mask] = other.weights[:, mask]
        self["_ll"][mask] = other.get_loglikelihood()[mask]
        self["_prev_inds"][:, mask] = other.previous_indices[:, mask]

        self["_mean"][mask] = other["_mean"][mask]
        self["_var"][mask] = other["_var"][mask]

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
