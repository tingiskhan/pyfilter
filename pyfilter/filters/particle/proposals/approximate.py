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

    def copy(self) -> "Proposal":
        return GaussianProposal()


# TODO: Clean up
class LinearizedGaussianProposal(GaussianProposal):
    """
    Implements a Gaussian proposal distribution based on a linearized version.
    """

    def __init__(self, num_steps: int = 1, alpha: float = 1e-3, use_second_order: bool = True):
        """
        Internal initializer for :class:`LinearizedGaussianProposal`.

        Args:
            num_steps (int, optional): number of steps.
            alpha (float, optional): alpha in gradient descent, only applies if ``use_second_order`` is ``True``.
            use_second_order (bool, optional): whether to use second order information.
        """

        super().__init__(None)
        self._n_steps = num_steps
        self._use_second_order = use_second_order
        self._alpha = alpha

    def sample_and_weight(self, y: torch.Tensor, state: ParticleFilterState, predictive_distribution: Distribution) -> Tuple[TimeseriesState, torch.Tensor]:
        unsqueeze_dim = -(self._model.hidden.n_dim + 1)
        mean_state = state.timeseries_state.copy(values=state.get_mean().unsqueeze(unsqueeze_dim))
        
        mean, std = self._model.hidden.mean_scale(mean_state)
        std = (state.get_variance().unsqueeze(unsqueeze_dim) + std.pow(2.0)).sqrt()

        kernel = find_mode_of_distribution(self._model, predictive_distribution, mean_state, mean, std, y, self._n_steps, self._alpha, self._use_second_order)
        
        x_result = state.timeseries_state.copy(values=kernel.expand(state.timeseries_state.batch_shape).sample)

        return x_result, self._weight_with_kernel(y, predictive_distribution, x_result, kernel)

    def copy(self) -> "Proposal":
        return LinearizedGaussianProposal(
            num_steps=self._n_steps,
            alpha=self._alpha,
            use_second_order=self._use_second_order
        )
