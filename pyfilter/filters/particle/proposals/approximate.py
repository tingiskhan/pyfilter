import torch
from stochproc.timeseries import TimeseriesState, AffineProcess
from typing import Tuple

from pyro.distributions import Distribution

from ..state import ParticleFilterState
from .base import Proposal
from .utils import find_mode_of_distribution, find_optimal_density


# TODO: Clean up? Perhaps move to separate modules for "exact" and approximate
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

    def sample_and_weight(self, y, state, predictive_distribution):
        unsqueeze_dim = -(self._model.hidden.n_dim + 1)
        mean_state = state.timeseries_state.copy(values=state.get_mean().unsqueeze(unsqueeze_dim))
        
        mean, std = self._model.hidden.mean_scale(mean_state)
        std = (state.get_variance().unsqueeze(unsqueeze_dim) + std.pow(2.0)).sqrt()

        kernel = find_mode_of_distribution(self._model, predictive_distribution, mean_state, mean, std, y, self._n_steps, self._alpha, self._use_second_order).expand(state.timeseries_state.batch_shape)        
        x_result = state.timeseries_state.copy(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, predictive_distribution, x_result, kernel)

    def copy(self) -> "Proposal":
        return LinearizedGaussianProposal(
            num_steps=self._n_steps,
            alpha=self._alpha,
            use_second_order=self._use_second_order
        )


class LinearGaussianProposal(GaussianProposal):
    # TODO: Fix doc

    def __init__(self, a_index: int = 0, b_index: int = None, s_index: int = -1, is_variance: bool = False):
        """
        Initializes the :class:`LinearGaussianProposal` class.

        Args:
            a_index: index of the parameter that constitutes :math:`A` in the observable process, assumes that
                it's the first one. If you pass ``None`` it is assumed that this corresponds to an identity matrix.
            b_index: index of the parameter that constitutes :math:`b` in the observable process.
            s_index: index of the parameter that constitutes :math:`s` in the observable process.
            is_variance: whether `s_index` parameter corresponds to a variance or standard deviation parameter.
        """

        super().__init__()
        self._a_index = a_index
        self._b_index = b_index
        self._s_index = s_index

        self._is_variance = is_variance

    def get_offset_and_scale(
        self, x: TimeseriesState, parameters: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standardizes the observation.

        Args:
            x: previous state.
            parameters: parameters of the observations process.
        """

        a_param = parameters[self._a_index]
        if self._b_index is None:
            return a_param, torch.tensor(0.0, device=a_param.device)

        return a_param, parameters[self._b_index]

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess):
            raise ValueError("Model combination not supported!")

        return super().set_model(model)

    def sample_and_weight(self, y, state, predictive_distribution):
        unsqueeze_dim = -(self._model.hidden.n_dim + 1)
        mean_state = state.timeseries_state.copy(values=state.get_mean().unsqueeze(unsqueeze_dim))

        mean, scale = self._model.hidden.mean_scale(mean_state)
        h_var_inv = (scale.pow(2.0) + state.get_variance().unsqueeze(unsqueeze_dim)).pow(-1.0)

        parameters = self._model.parameters
        a_param, offset = self.get_offset_and_scale(mean_state, parameters)
        o_var_inv = parameters[self._s_index].pow(-2.0 if not self._is_variance else -1.0)

        kernel = find_optimal_density(y - offset, mean, h_var_inv, o_var_inv, a_param, self._model).expand(state.timeseries_state.batch_shape)
        x_result = state.timeseries_state.propagate_from(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, predictive_distribution, x_result, kernel)

    def copy(self) -> "Proposal":
        return LinearGaussianProposal(self._a_index, self._b_index, self._s_index, self._is_variance)
