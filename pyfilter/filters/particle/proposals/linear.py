from typing import Tuple

import torch
from stochproc.timeseries import AffineProcess, LinearStateSpaceModel, TimeseriesState
from torch.distributions import MultivariateNormal, Normal
from torch.linalg import cholesky_ex

from ....utils import construct_diag_from_flat
from .base import Proposal
from .utils import find_optimal_density


class LinearGaussianObservations(Proposal):
    r"""
    Implements the optimal proposal density whenever we have that both the latent and observation densities are
    Gaussian, and that the mean of the observation density can be expressed as a linear combination of the latent
    states. More specifically, we have that
        .. math::
            Y_t = b + A \cdot X_t + V_t, \newline
            X_{t+1} = f_\theta(X_t) + g_\theta(X_t) W_{t+1},

    where :math:`A` is a matrix of dimension ``{observation dimension, latent dimension}``,
    :math:`V_t` and :math:`W_t` two independent zero mean and unit variance Gaussians, and :math:`\theta` denotes the
    parameters of the functions :math:`f` and :math:`g` (excluding :math:`X_t`).
    """

    def _get_offset_and_scale(
        self, x: TimeseriesState, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return a, b

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess) or not isinstance(model, LinearStateSpaceModel):
            raise ValueError("Model combination not supported!")

        return super().set_model(model)

    def sample_and_weight(self, y, prediction):
        x = prediction.get_timeseries_state()

        mean, scale = self._model.hidden.mean_scale(x)
        x_dist = self._model.hidden.build_density(x)

        h_var_inv = scale.pow(-2.0)

        x_copy = x.copy(values=mean)

        a, b, s = self._model.parameters
        a, offset = self._get_offset_and_scale(x, a, b)
        o_var_inv = s.pow(-2.0)

        kernel = find_optimal_density(y - offset, mean, h_var_inv, o_var_inv, a, self._model)
        x_result = x_copy.propagate_from(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, x_dist, x_result, kernel)

    def pre_weight(self, y, x):
        _, h_scale = self._model.hidden.mean_scale(x)

        h_var = h_scale.pow(2.0)

        a, b, s = self._model.parameters
        a, offset = self._get_offset_and_scale(x, a, b)
        o_var = s.pow(2.0)

        if self._model.hidden.n_dim == 0:
            a = a.unsqueeze(-1)

        obs_is_1d = self._model.n_dim == 0

        a_unsqueezed = a if not obs_is_1d else a.unsqueeze(-2)
        a_transposed = a_unsqueezed.transpose(-2, -1)

        diag_h_var = construct_diag_from_flat(h_var, self._model.hidden.event_shape)
        diag_o_var = construct_diag_from_flat(o_var, self._model.event_shape)

        cov = diag_o_var + a_unsqueezed.matmul(diag_h_var).matmul(a_transposed)

        if obs_is_1d:
            o_loc = offset + a.squeeze(-1) * x.value
            kernel = Normal(o_loc, cov[..., 0, 0].sqrt())
        else:
            o_loc = offset + (a_unsqueezed @ x.value.unsqueeze(-1)).squeeze(-1)
            kernel = MultivariateNormal(o_loc, scale_tril=cholesky_ex(cov)[0])

        return kernel.log_prob(y)

    def copy(self) -> "Proposal":
        return LinearGaussianObservations()
