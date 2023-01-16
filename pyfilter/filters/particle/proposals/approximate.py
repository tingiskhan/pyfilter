import torch

from ..state import ParticleFilterPrediction
from .base import Proposal
from .linearized import Linearized
from .linear import LinearGaussianObservations
from .utils import find_optimal_density
from ..utils import get_filter_mean_and_variance


# TODO: Fix GPF proposals shapes, a bit whacky
class GaussianProposal(Proposal):
    """
    Implements a proposal distribution based on a Gaussian approximation.
    """

    # TODO: Perhaps rename?
    def sample_and_weight(self, y: torch.Tensor, prediction: ParticleFilterPrediction):
        predictive_distribution = prediction.get_predictive_density(self._model, approximate=True)

        timeseries_state = prediction.get_timeseries_state()

        predictive_distribution = predictive_distribution.expand(timeseries_state.batch_shape)
        x_vals = predictive_distribution.sample()

        x_result = timeseries_state.copy(values=x_vals)

        observation_density = self._model.build_density(x_result)
        w = observation_density.log_prob(y)

        return x_result, w

    def copy(self) -> "Proposal":
        return GaussianProposal()


class GaussianLinearized(Linearized):
    """
    Same as :class:`Linearized`, but in which we use an approximation of the predictive density.
    """

    def sample_and_weight(self, y, prediction):
        timeseries_state = prediction.get_timeseries_state()
        predictive_mean, predictive_variance = get_filter_mean_and_variance(
            timeseries_state, prediction.normalized_weights, keep_dim=False
        )

        predictive_mean.unsqueeze_(0)
        predictive_variance.unsqueeze_(0)

        # TODO: Figure out broadcasting
        mean_state = timeseries_state.copy(values=predictive_mean)

        mean, std = self._model.hidden.mean_scale(mean_state)
        std = (predictive_variance + std.pow(2.0)).sqrt()

        initial_state = mean_state.propagate_from(values=mean.clone())

        predictive_distribution = prediction.get_predictive_density(self._model, approximate=True)
        kernel = self._mode_finder.find_mode_legacy(predictive_distribution, initial_state, std, y)

        x_result = timeseries_state.copy(values=kernel.expand(timeseries_state.batch_shape).sample)

        return x_result, self._weight_with_kernel(y, predictive_distribution, x_result, kernel)

    def copy(self) -> "Proposal":
        return GaussianLinearized(n_steps=self._n_steps, alpha=self._alpha, use_second_order=self._use_second_order)


# TODO: Horrible name...
class GaussianLinear(LinearGaussianObservations):
    """
    Same as :class:`LinearGaussianObservations`, but in which we use an approximation of the predictive density.
    """

    def sample_and_weight(self, y, prediction):
        timeseries_state = prediction.get_timeseries_state()
        predictive_mean, predictive_variance = get_filter_mean_and_variance(
            timeseries_state, prediction.normalized_weights, keep_dim=False
        )

        predictive_mean.unsqueeze_(0)
        predictive_variance.unsqueeze_(0)

        mean_state = timeseries_state.copy(values=predictive_mean)

        mean, scale = self._model.hidden.mean_scale(mean_state)
        h_var_inv = (scale.pow(2.0) + predictive_variance).pow(-1.0)

        a, b, s = self._model.transformed_parameters()
        a, offset = self._get_offset_and_scale(mean, a, b)
        o_var_inv = s.pow(-2.0)

        kernel = find_optimal_density(y - offset, mean, h_var_inv, o_var_inv, a, self._model).expand(
            timeseries_state.batch_shape
        )
        x_result = timeseries_state.propagate_from(values=kernel.sample)

        predictive_distribution = prediction.get_predictive_density(self._model, approximate=True)

        return x_result, self._weight_with_kernel(y, predictive_distribution, x_result, kernel)

    def copy(self) -> "Proposal":
        return GaussianLinear()
