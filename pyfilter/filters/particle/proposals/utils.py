import torch

from stochproc.timeseries import StateSpaceModel, TimeseriesState
from pyro.distributions import Distribution, Normal
from torch.autograd import grad


# TODO: Clean this up...
def find_mode_of_distribution(model: StateSpaceModel, x_dist: Distribution, initial_state: TimeseriesState, mean: torch.Tensor, std: torch.Tensor, y: torch.Tensor, n_steps: int = 1, alpha: float = 1e-3, use_second_order = False) -> Distribution:
    """
    Finds the mode of the joint distribution ... .

    Returns:
        Distribution: 
    """

    mode_state = initial_state.copy(values=mean)

    for _ in range(n_steps):
        mean.requires_grad_(True)

        y_dist = model.build_density(mode_state)
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
        mode_state = mode_state.copy(values=mean)

    kernel = Normal(mean, std)
    if model.hidden.event_shape.numel() > 1:
        kernel = kernel.to_event(1)
    
    return kernel
