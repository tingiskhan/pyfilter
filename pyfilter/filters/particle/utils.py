import torch
from typing import Tuple

from stochproc.timeseries import TimeseriesState


def log_likelihood(importance_weights: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Given the importance weights of a particle filter, return an estimate of the log likelihood.

    Args:
        importance_weights: the importance weights of the particle filter.
        weights: optional parameter specifying whether the weights associated with the importance weights, used in
            :class:`~pyfilter.filters.particle.APF`.
    """

    max_w, _ = importance_weights.max(-1)

    if weights is None:
        temp = (importance_weights - (max_w.unsqueeze(-1) if max_w.dim() > 0 else max_w)).exp().mean(-1).log()
    else:
        temp = (
            (weights * (importance_weights - (max_w.unsqueeze(-1) if max_w.dim() > 0 else max_w)).exp()).sum(-1).log()
        )

    return max_w + temp


# TODO: Does not work yet, fix
def get_filter_mean_and_variance(state: TimeseriesState, weights: torch.Tensor, covariance: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the filtered mean and variance given a weighted particle set.

    Args:
        state (TimeseriesState): state of timeseries for which to get mean and variance.
        weights (torch.Tensor): normalized weights.
        covariance (bool): whether to calculate the covariance or just variance.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
    """

    values = state.value
    if not state.event_shape:
        values = values.unsqueeze(-1)

    weights = weights.unsqueeze(-2)
    mean = weights @ values

    # TODO: This change brings about filter means to have an extra dimension
    # TODO: Think about whether the unsqueeze should be kept...
    centered = values - mean

    if not covariance or not state.event_shape:
        var = (weights @ centered.pow(2.0)).squeeze(-2)
    else:
        centered = values - mean
        covariances = centered.unsqueeze(-1) @ centered.unsqueeze(-2)
 
        var = torch.einsum("...b,...bij -> ...ij", weights.squeeze(-2), covariances)

    return mean.squeeze(-2), var
