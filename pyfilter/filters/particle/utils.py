import torch


def loglikelihood(w: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Given the importance weights of a particle filter, return an estimate of the log likelihood.

    Args:
        w: The importance weights of the particle filter.
        weights: Optional parameter, the weights associated with the importance weights. Used by APF.
    """

    maxw, _ = w.max(-1)

    if weights is None:
        temp = torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw)).mean(-1).log()
    else:
        temp = (weights * torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw))).sum(-1).log()

    return maxw + temp