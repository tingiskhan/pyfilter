import torch
from typing import Tuple

from stochproc.timeseries import TimeseriesState


def log_likelihood(importance_weights: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Given the importance weights of a particle filter, return an estimate of the log likelihood.

    Args:
        importance_weights (torch.Tensor): importance weights of the particle filter.
        weights (torch.Tensor): optional parameter specifying whether the weights associated with the importance weights.
    """

    max_w, _ = importance_weights.max(dim=0)

    temp = (importance_weights - max_w).exp()
    if weights is None:
        weights = 1.0 / importance_weights.shape[0]

    return max_w + (weights * temp).sum(dim=0).log()


# TODO: The keep_dim is a tad bit weird?
def get_filter_mean_and_variance(
    state: TimeseriesState, weights: torch.Tensor, covariance: bool = False, keep_dim: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Gets the filtered mean and variance given a weighted particle set.

    Args:
        state (TimeseriesState): state of timeseries for which to get mean and variance.
        weights (torch.Tensor): normalized weights.
        covariance (bool): whether to calculate the covariance or just variance.
        keep_dim (bool): whether to keep last dimension.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: returns the tuple ``{mean, variance}``.
    """

    values = state.value
    if not state.event_shape:
        values = values.unsqueeze(-1)

    weights = weights.unsqueeze(-1)
    mean = (weights * values).sum(dim=0)

    # TODO: This change brings about filter means to have an extra dimension
    # TODO: Think about whether the unsqueeze should be kept...
    centered = values - mean

    if not covariance or not state.event_shape:
        var = (weights * centered.pow(2.0)).sum(dim=0)

        if not keep_dim:
            var.squeeze_(-1)
    else:
        covariances = centered.unsqueeze(-1) @ centered.unsqueeze(-2)
        var = torch.einsum("b..., b...ij -> ...ij", weights.squeeze(-1), covariances)

    if not keep_dim:
        mean.squeeze_(-1)

    return mean, var
