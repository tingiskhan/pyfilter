from typing import OrderedDict
import torch

from stochproc.timeseries import StateSpaceModel, TimeseriesState
from pyro.distributions import Distribution, Normal, MultivariateNormal
from torch.linalg import cholesky_ex
from functorch import vmap, hessian, grad
from torch.autograd import grad as agrad

from ....utils import construct_diag_from_flat


def _infer_shapes(parameters, i):
    from ....inference.parameter import PriorBoundParameter

    if isinstance(parameters, tuple):
        # NB: We always assume that the parameters can only be batched in one dimension
        return tuple(None if not isinstance(p, PriorBoundParameter) else (0 if i == 0 else None) for p in parameters)
    elif not isinstance(parameters, dict):
        raise NotImplementedError(f"Does not support {parameters.__class__}")

    # TODO: This must be identical to `parameters`...
    res = OrderedDict([])
    for k, v in parameters.items():
        res[k] = _infer_shapes(v, i)

    return res


class ModeFinder(object):
    """
    Implements a class for finding the mode of the joint distribution of a state space model.
    """

    def __init__(self, model: StateSpaceModel, num_steps: int, alpha: float = 1e-2, use_hessian: bool = False):
        """
        Internal initializer for :class:`ModeFinder`.

        Args:
            model (StateSpaceModel): model to use when building distribution.
            num_steps (int): number of steps to take in order to find model.
            alpha (float, optional): when :attr:`use_hessian` is False we use gradient descent with :attr:`alpha` as step size. Defaults to 1e-3.
            use_hessian (bool, optional): whether to use Hessian information. Defaults to False.
        """

        self._model = model
        self._num_steps = num_steps
        self._alpha = alpha
        self._use_hessian = use_hessian

        self._grad_fun = None
        self._hess_fun = None

        self.initialized = False

    def initialize(self, batch_shape: torch.Size):
        """
        Initializes the class for the first time.

        Args:
            batch_shape (torch.Size): batch shape to consider.
        """

        grad_fun = grad(self._model_likelihood)
        # TODO: Consider using second order without Hessian as we seem to have some issues with memory consumption
        hess_fun = hessian(self._model_likelihood)

        for i, _ in enumerate(batch_shape):
            parameter_dims = {
                "hidden": _infer_shapes(self._model.hidden.yield_parameters()["parameters"], i),
                "ssm": _infer_shapes(self._model.yield_parameters()["parameters"], i),
            }

            in_dims = (0, None, 0, parameter_dims, None, None)

            grad_fun = vmap(grad_fun, in_dims=in_dims)
            hess_fun = vmap(hess_fun, in_dims=in_dims)

        self._grad_fun = grad_fun
        self._hess_fun = hess_fun

        self.initialized = True

    @staticmethod
    def _model_likelihood(
        x: torch.Tensor,
        y: torch.Tensor,
        old_x: torch.Tensor,
        parameters,
        state: TimeseriesState,
        model: StateSpaceModel,
    ) -> torch.Tensor:
        with model.hidden.override_parameters(parameters["hidden"]), model.override_parameters(parameters["ssm"]):
            x_dist = model.hidden.build_density(state.copy(values=old_x))
            y_dist = model.build_density(state.propagate_from(values=x))

            return y_dist.log_prob(y) + x_dist.log_prob(x)

    # TODO: Something is wonky with second order information for models with multiple observations
    def find_mode(
        self, prev_x: torch.Tensor, x: torch.Tensor, y: torch.Tensor, std: torch.Tensor, dummy_state: TimeseriesState
    ) -> Distribution:
        """
        Finds the mode of the state space model's joint distribution.

        Args:
            prev_x (torch.Tensor): previous particles.
            x (torch.Tensor): particles around which to linearize.
            y (torch.Tensor): current observation.
            std (torch.Tensor): standard deviation of the linearized state.
            dummy_state (TimeseriesState): dummy state used in loglikelihood function.

        Returns:
            Distribution: returns a proposal distribution.
        """

        y_squeezed = y.squeeze(-1)
        x = x.clone()
        fill_std = std.clone()

        step = self._alpha

        parameters = {
            "hidden": self._model.hidden.yield_parameters()["parameters"],
            "ssm": self._model.yield_parameters()["parameters"],
        }

        for _ in range(self._num_steps):
            gradient = self._grad_fun(x, y_squeezed, prev_x, parameters, dummy_state, self._model)

            if self._use_hessian:
                hess = self._hess_fun(x, y_squeezed, prev_x, parameters, dummy_state, self._model)

                if self._model.hidden.n_dim == 0:
                    d_h = (2.0 * hess).clip(min=0.0)
                    cov = -(hess - d_h).pow(-1).squeeze(-1)

                    step = cov * gradient
                    std = cov.sqrt()
                else:
                    lamda_min = torch.linalg.eigvalsh(hess).real.min(dim=-1).values

                    # TODO: Can we replace this...?
                    d_h = (2.0 * lamda_min).clip(min=0.0).view(*hess.shape[:-2], 1, 1) * torch.eye(hess.shape[-1])
                    cov = -torch.linalg.pinv(hess - d_h)

                    step = (cov @ gradient.unsqueeze(-1)).squeeze(-1)
                    std = torch.linalg.cholesky_ex(cov)[0]

            x += step

        if self._model.hidden.n_dim == 0:
            return Normal(x, std)

        if not self._use_hessian:
            return Normal(x, std).to_event(1)

        return MultivariateNormal(x, scale_tril=std)

    def find_mode_legacy(
        self, x_dist: Distribution, initial_state: TimeseriesState, std: torch.Tensor, y: torch.Tensor
    ) -> Distribution:
        """
        Finds the mode of the state space model's joint distribution using `torch.autograd` rather than `functorch`.

        Args:
            x_dist (Distribution): predictive distribution of the latent process.
            initial_state (TimeseriesState): state of the timeseries.
            std (torch.Tensor): initial standard deviation to linearize around.
            y (torch.Tensor): observation.

        Returns:
            Distribution: returns a normal approximation of the density utilizing the mode.
        """

        fill_mean = initial_state.value.clone()
        fill_std = std.clone()

        mean = initial_state.value
        step = self._alpha

        for _ in range(self._num_steps):
            mean.requires_grad_(True)

            y_dist = self._model.build_density(initial_state)
            logl = y_dist.log_prob(y) + x_dist.log_prob(mean)
            gradient = agrad(logl, mean, grad_outputs=torch.ones_like(logl), create_graph=self._use_hessian)[-1]

            if self._use_hessian:
                diag_hess = agrad(gradient, mean, grad_outputs=torch.ones_like(gradient))[-1]

                if self._model.n_dim > 0:
                    v = diag_hess.min(dim=-1).values.unsqueeze(-1)
                else:
                    v = diag_hess

                dh = (2 * v).clip(min=0.0)

                step = (-diag_hess + dh).pow(-1)
                std = step.sqrt()

                gradient = gradient.detach()

            mean = mean.detach()
            mean += step * gradient
            initial_state = initial_state.copy(values=mean)

        # TODO: Use masked scatter
        mask = ~(mean.isfinite() & std.isfinite())

        mean[mask] = fill_mean[mask]
        std[mask] = fill_std[mask]

        kernel = Normal(mean, std)
        if self._model.hidden.n_dim > 0:
            kernel = kernel.to_event(1)

        return kernel


def find_optimal_density(
    y: torch.Tensor,
    loc: torch.Tensor,
    h_var_inv: torch.Tensor,
    o_var_inv: torch.Tensor,
    c: torch.Tensor,
    model: StateSpaceModel,
) -> Distribution:
    """
    Finds the optimal proposal distribution for particle filters.

    Args:
        y (torch.Tensor): (de-meaned) observation.
        loc (torch.Tensor): location of hidden process.
        h_var_inv (torch.Tensor): inverse variance of the hidden process.
        o_var_inv (torch.Tensor): inverse variance of the observable process.
        c (torch.Tensor): matrix of observation.
        model (StateSpaceModel): the underlying state space model.

    Returns:
        Distribution: returns the optimal density.
    """

    hidden_is_1d = model.hidden.n_dim == 0
    obs_is_1d = model.n_dim == 0

    if hidden_is_1d:
        c = c.unsqueeze(-1)

    c_unsqueezed = c if not obs_is_1d else c.unsqueeze(-2)

    c_transposed = c_unsqueezed.transpose(-2, -1)
    o_inv_cov = construct_diag_from_flat(o_var_inv, model.event_shape)
    t_2 = c_transposed.matmul(o_inv_cov).matmul(c_unsqueezed)

    cov = (construct_diag_from_flat(h_var_inv, model.hidden.event_shape) + t_2).inverse()
    t_1 = h_var_inv * loc

    if hidden_is_1d:
        t_1 = t_1.unsqueeze(-1)

    t_2 = o_inv_cov.squeeze(-1) * y.unsqueeze(-1) if obs_is_1d else o_inv_cov.matmul(y)
    t_3 = c_transposed.matmul(t_2.unsqueeze(-1))
    mean = cov.matmul(t_1.unsqueeze(-1) + t_3).squeeze(-1)

    if hidden_is_1d:
        return Normal(mean.squeeze(-1), cov[..., 0, 0].sqrt())

    return MultivariateNormal(mean, scale_tril=cholesky_ex(cov)[0])
