import torch

from stochproc.timeseries import StateSpaceModel, TimeseriesState
from pyro.distributions import Distribution, Normal, MultivariateNormal
from torch.linalg import cholesky_ex
from torch.autograd import grad

from ....utils import construct_diag_from_flat


# TODO: Clean this up...
def find_mode_of_distribution(model: StateSpaceModel, x_dist: Distribution, initial_state: TimeseriesState, std: torch.Tensor, y: torch.Tensor, n_steps: int = 1, alpha: float = 1e-3, use_second_order = False) -> Distribution:
    """
    Finds the mode of the joint distribution given by ``model``.

    Args:
        model (StateSpaceModel): underlying state space model.
        x_dist (Distribution): predictive distribution of the latent process.
        initial_state (TimeseriesState): state of the timeseries.
        std (torch.Tensor): initial standard deviation to linearize around.
        y (torch.Tensor): observation.
        n_steps (int, optional): number of linearization steps to perform. Defaults to 1.
        alpha (float, optional): if ``use_second_order=False`` utilizes gradient descent, ``alpha`` is the step size. Defaults to 1e-3.
        use_second_order (bool, optional): whether to use second order information. Defaults to False.

    Returns:
        Distribution: returns a normal approximation of the optimal density utilizing the mode.
    """

    mean = initial_state.value

    for _ in range(n_steps):
        mean.requires_grad_(True)

        y_dist = model.build_density(initial_state)
        logl = y_dist.log_prob(y) + x_dist.log_prob(mean)
        gradient = grad(logl, mean, grad_outputs=torch.ones_like(logl), create_graph=use_second_order)[-1]

        ones_like_g = torch.ones_like(gradient)
        step = alpha * ones_like_g
        if use_second_order:
            neg_inv_hess = -grad(gradient, mean, grad_outputs=ones_like_g)[-1].pow(-1.0)

            # TODO: There is a better approach in Dahlin, find it
            mask = neg_inv_hess > 0.0

            step[mask] = neg_inv_hess[mask]
            std[mask] = neg_inv_hess[mask].sqrt()

            gradient = gradient.detach()

        mean = mean.detach()
        mean += step * gradient
        initial_state = initial_state.copy(values=mean)

    kernel = Normal(mean, std.nan_to_num(alpha))
    if model.hidden.event_shape.numel() > 1:
        kernel = kernel.to_event(1)
    
    return kernel


def find_optimal_density(y: torch.Tensor, loc: torch.Tensor, h_var_inv: torch.Tensor, o_var_inv: torch.Tensor, c: torch.Tensor, model: StateSpaceModel) -> Distribution:
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
