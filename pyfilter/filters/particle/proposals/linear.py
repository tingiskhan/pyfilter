from .base import Proposal
from torch.linalg import cholesky_ex
from torch.distributions import Normal, MultivariateNormal
from stochproc.timeseries import AffineProcess
from ....utils import construct_diag_from_flat


class LinearGaussianObservations(Proposal):
    r"""
    Implements the optimal proposal density whenever we have that both the latent and observation densities are
    Gaussian, and that the mean of the observation density can be expressed as a linear combination of the latent
    states. More specifically, we have that
        .. math::
            Y_t = A \cdot X_t + V_t, \newline
            X_{t+1} = f_\theta(X_t) + g_\theta(X_t) W_{t+1},

    where :math:`A` is a matrix of dimension ``(observation dimension, latent dimension)``,
    :math:`V_t` and :math:`W_t` two independent zero mean and unit variance Gaussians, and :math:`\theta` denotes the
    parameters of the functions :math:`f` and :math:`g` (excluding :math:`X_t`).
    """

    def __init__(self, parameter_index: int):
        """
        Initializes the :class:`LinearGaussianObservations` class.
        """

        super().__init__()
        self._hidden_is1d = None
        self._parameter_index = parameter_index

    def set_model(self, model):
        if not isinstance(model.hidden, AffineProcess):
            raise ValueError("Model combination not supported!")

        self._model = model
        self._hidden_is1d = self._model.hidden.n_dim == 0

        return self

    def _kernel_1d(self, y, loc, h_var_inv, o_var_inv, c, observable_is_1d):
        cov = 1 / (h_var_inv + c ** 2 * o_var_inv)
        m = cov * (h_var_inv * loc + c * o_var_inv * y)

        return Normal(m, cov.sqrt())

    def _kernel_2d(self, y, loc, h_var_inv, o_var_inv, c, obs_n_dim):
        observable_is_1d = obs_n_dim == 0
        tc = c if not observable_is_1d else c.unsqueeze(-2)

        ttc = tc.transpose(-2, -1)
        o_inv_cov = construct_diag_from_flat(o_var_inv, obs_n_dim)
        t2 = ttc.matmul(o_inv_cov).matmul(tc)

        cov = (construct_diag_from_flat(h_var_inv, self._model.hidden.n_dim) + t2).inverse()
        t1 = h_var_inv * loc
        t2 = o_inv_cov.squeeze(-1) * y if observable_is_1d else o_inv_cov.matmul(y)

        t3 = ttc.matmul(t2.unsqueeze(-1)).squeeze(-1)
        m = cov.matmul((t1 + t3).unsqueeze(-1)).squeeze(-1)

        return MultivariateNormal(m, scale_tril=cholesky_ex(cov)[0])

    def sample_and_weight(self, y, x):
        mean, scale = self._model.hidden.mean_scale(x)
        x_dist = self._model.hidden.build_density(x)

        h_var_inv = scale.pow(-2.0)

        x_copy = x.copy(values=mean)
        observable_dist = self._model.build_density(x_copy)
        o_var_inv = observable_dist.variance.pow(-1.0)

        observable_parameters = self._model.functional_parameters()

        kernel_func = self._kernel_1d if self._hidden_is1d else self._kernel_2d
        kernel = kernel_func(
            y,
            mean,
            h_var_inv,
            o_var_inv,
            observable_parameters[self._parameter_index],
            len(observable_dist.event_shape),
        )

        x_result = x_copy.propagate_from(values=kernel.sample)

        return x_result, self._weight_with_kernel(y, x_dist, x_result, kernel)

    def pre_weight(self, y, x):
        h_loc, h_scale = self._model.hidden.mean_scale(x)
        new_state = x.propagate_from(values=h_loc)

        observable_parameters = self._model.functional_parameters()

        h_var = h_scale.pow(2.0)
        obs_dist = self._model.build_density(new_state)

        o_loc, o_var = obs_dist.mean, obs_dist.variance
        c = observable_parameters[self._parameter_index]

        obs_n_dim = len(obs_dist.event_shape)

        if obs_n_dim == 0:
            if self._hidden_is1d:
                cov = o_var + c ** 2 * h_var
            else:
                tc = c.unsqueeze(-2)
                diag_h_var = construct_diag_from_flat(h_var, self._model.hidden.n_dim)
                cov = o_var + tc.matmul(diag_h_var).matmul(tc.transpose(-2, -1))[..., 0, 0]

            return Normal(o_loc, cov.sqrt()).log_prob(y)

        if self._hidden_is1d:
            tc = c.unsqueeze(-2)
            cov = (o_var + tc.matmul(tc.transpose(-2, -1)) * h_var)[..., 0, 0]
        else:
            diag_o_var = construct_diag_from_flat(o_var, obs_n_dim)
            diag_h_var = construct_diag_from_flat(h_var, self._model.hidden.n_dim)
            cov = diag_o_var + c.matmul(diag_h_var).matmul(c.transpose(-2, -1))

        return MultivariateNormal(o_loc, cov).log_prob(y)

    def copy(self) -> "Proposal":
        return LinearGaussianObservations(self._parameter_index)
