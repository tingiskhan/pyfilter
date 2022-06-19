import torch


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
