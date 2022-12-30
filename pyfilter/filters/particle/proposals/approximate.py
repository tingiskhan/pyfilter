import torch
from stochproc.timeseries import TimeseriesState
from typing import Tuple

from pyro.distributions import Distribution

from ..state import ParticleFilterState
from .base import Proposal
from .utils import find_mode_of_distribution


class GaussianProposal(Proposal):
    """
    Implements a proposal distribution based on a Gaussian approximation.
    """

    # TODO: Perhaps rename?
    def sample_and_weight(self, y: torch.Tensor, state: ParticleFilterState, predictive_distribution: Distribution) -> Tuple[TimeseriesState, torch.Tensor]:
        predictive_distribution = predictive_distribution.expand(state.timeseries_state.batch_shape)
        x_vals = predictive_distribution.sample()

        x_result = state.timeseries_state.copy(values=x_vals)
  
        observation_density = self._model.build_density(x_result)
        w = observation_density.log_prob(y)

        return x_result, w


# TODO: Clean up
class LinearizedGaussianProposal(GaussianProposal):
    """
    Implements a Gaussian proposal distribution based on a linearized version.
    """

    def __init__(self, num_steps: int = 5, alpha: float = 1e-3):
        """
        Internal initializer for :class:`LinearizedGaussianProposal`.

        Args:
            num_steps (int, optional): number of steps. Defaults to 5.
            alpha (float, optional): _description_. Defaults to 1e-3.
        """

        super().__init__(None)
        self._n_steps = num_steps
        self._use_second_order = True
        self._alpha = alpha
        self._is_1d = None

    def sample_and_weight(self, y: torch.Tensor, state: ParticleFilterState, predictive_distribution: Distribution) -> Tuple[TimeseriesState, torch.Tensor]:
        mean_state = state.timeseries_state.copy(values=state.get_mean())
        
        mean, std = self._model.hidden.mean_scale(mean_state)
        std = (state.get_variance() + std.pow(2.0)).sqrt()

        kernel = find_mode_of_distribution(self._model, predictive_distribution, mean_state, mean, std, y, self._n_steps, self._alpha, self._use_second_order)
        
        x_result = state.timeseries_state.copy(values=kernel.sample(state.timeseries_state.batch_shape))

        return x_result, self._weight_with_kernel(y, predictive_distribution, x_result, kernel)
