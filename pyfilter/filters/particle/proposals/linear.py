from typing import Tuple

import torch
from torch.linalg import cholesky_ex
from torch.distributions import Normal, MultivariateNormal
from stochproc.timeseries import AffineProcess, TimeseriesState

from .base import Proposal
from ....utils import construct_diag_from_flat


class LinearGaussianObservations(Proposal):
    r"""
    Implements the optimal proposal density whenever we have that both the latent and observation densities are
    Gaussian, and that the mean of the observation density can be expressed as a linear combination of the latent
    states. More specifically, we have that
        .. math::
            Y_t = b + A \cdot X_t + V_t, \newline
            X_{t+1} = f_\theta(X_t) + g_\theta(X_t) W_{t+1},

    where :math:`A` is a matrix of dimension ``(observation dimension, latent dimension)``,
    :math:`V_t` and :math:`W_t` two independent zero mean and unit variance Gaussians, and :math:`\theta` denotes the
    parameters of the functions :math:`f` and :math:`g` (excluding :math:`X_t`).
    """

    def __init__(self, a_index: int = 0, b_index: int = None, s_index: int = -1, is_variance: bool = False):
        """
        Initializes the :class:`LinearGaussianObservations` class.

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

        self._hidden_is1d = None
        self._obs_is1d = None

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
            return a_param, 0.0

        return a_param, parameters[self._b_index]

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess):
            raise ValueError("Model combination not supported!")

        self._model = model
        self._hidden_is1d = self._model.hidden.n_dim == 0
        self._obs_is1d = self._model.n_dim == 0

        return self

    def _kernel(self, y, loc, h_var_inv, o_var_inv, c):
        if self._hidden_is1d:
            c = c.unsqueeze(-1)

        c_ = c if not self._obs_is1d else c.unsqueeze(-2)

        c_transposed = c_.transpose(-2, -1)
        o_inv_cov = construct_diag_from_flat(o_var_inv, self._model.event_shape)
        t2 = c_transposed.matmul(o_inv_cov).matmul(c_)

        cov = (construct_diag_from_flat(h_var_inv, self._model.hidden.event_shape) + t2).inverse()
        t1 = h_var_inv * loc

        if self._hidden_is1d:
            t1 = t1.unsqueeze(-1)

        t2 = o_inv_cov.squeeze(-1) * y.unsqueeze(-1) if self._obs_is1d else o_inv_cov.matmul(y)
        t3 = c_transposed.matmul(t2.unsqueeze(-1))
        m = cov.matmul(t1.unsqueeze(-1) + t3).squeeze(-1)

        if self._hidden_is1d:
            return Normal(m.squeeze(-1), cov[..., 0, 0].sqrt())

        return MultivariateNormal(m, scale_tril=cholesky_ex(cov)[0])

    def sample_and_weight(self, y, x):
        mean, scale = self._model.hidden.mean_scale(x)
        x_dist = self._model.hidden.build_density(x)

        h_var_inv = scale.pow(-2.0)

        x_copy = x.copy(values=mean)

        parameters = self._model.functional_parameters()
        a_param, offset = self.get_offset_and_scale(x, parameters)
        o_var_inv = parameters[self._s_index].pow(-2.0 if not self._is_variance else -1.0)

        kernel = self._kernel(y - offset, mean, h_var_inv, o_var_inv, a_param)
        x_result = x_copy.propagate_from(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, x_dist, x_result, kernel)

    def pre_weight(self, y, x):
        h_loc, h_scale = self._model.hidden.mean_scale(x)

        h_var = h_scale.pow(2.0)

        observable_parameters = self._model.functional_parameters()
        c, offset = self.get_offset_and_scale(x, observable_parameters)
        o_var = observable_parameters[self._s_index].pow(2.0 if not self._is_variance else 1.0)

        if self._hidden_is1d:
            c = c.unsqueeze(-1)

        c_ = c if not self._obs_is1d else c.unsqueeze(-2)
        c_transposed = c_.transpose(-2, -1)

        diag_h_var = construct_diag_from_flat(h_var, self._model.hidden.event_shape)
        diag_o_var = construct_diag_from_flat(o_var, self._model.event_shape)

        cov = diag_o_var + c_.matmul(diag_h_var).matmul(c_transposed)

        if self._obs_is1d:
            o_loc = offset + c.squeeze() * x.values
            kernel = Normal(o_loc, cov[..., 0, 0].sqrt())
        else:
            o_loc = offset + (c_ @ x.values.unsqueeze(-1)).squeeze(-1)
            kernel = MultivariateNormal(o_loc, scale_tril=cholesky_ex(cov)[0])

        return kernel.log_prob(y)

    def copy(self) -> "Proposal":
        return LinearGaussianObservations(self._a_index, self._b_index, self._s_index, is_variance=self._is_variance)
