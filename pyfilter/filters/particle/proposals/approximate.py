import torch
from stochproc.timeseries import TimeseriesState
from typing import Tuple

from pyro.distributions import Normal, MultivariateNormal

from ..state import ParticleFilterState
from .base import Proposal


class GaussianProposal(Proposal):
    """
    Implements a proposal distribution based on a Gaussian approximation.
    """

    def _generate_importance_density(self, state: ParticleFilterState):
        dim = -(self._model.hidden.n_dim + 1)
        mean = state.get_mean().unsqueeze(dim)
        var = state.get_variance().unsqueeze(dim) if dim == -1 else state.get_covariance().unsqueeze(dim - 1)

        if self._model.hidden.n_dim == 0:
            dist = Normal(mean, var.sqrt())
        else:
            dist = MultivariateNormal(mean, covariance_matrix=var)
        
        return dist.expand(state.timeseries_state.batch_shape)

    # TODO: Fix such that all proposals take particlefilterstate as input?
    def sample_and_weight(self, y: torch.Tensor, x: ParticleFilterState) -> Tuple[TimeseriesState, torch.Tensor]:
        density = self._generate_importance_density(x)
        x_vals = density.sample()

        x_copy = x.timeseries_state.copy(values=x_vals)
        x_new = self._model.hidden.propagate(x_copy)

        observation_density = self._model.build_density(x_new)
        w = observation_density.log_prob(y)

        return x_new, w
