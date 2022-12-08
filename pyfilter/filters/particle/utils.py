import torch
from stochproc.timeseries import StructuralStochasticProcess


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


class Unsqueezer(object):
    """
    Helper object for temporarily squeezing/unsqueezing parameters.
    """

    def __init__(self, dim_to_unsqueeze: int, module: StructuralStochasticProcess, do_unsqueeze: bool):
        """
        Initializes :class:`Unsqueezer`.

        Args:
            dim_to_unsqueeze: dimension to unsqueeze.
            module: module to keep track of.
            do_unsqueeze: whether to perform an unsqueeze operation.
        """

        from ...inference import PriorBoundParameter

        self.dim_to_unsqueeze = dim_to_unsqueeze
        self.params = tuple(
            p
            for p in module.parameters + module.initial_parameters
            if isinstance(p, PriorBoundParameter) and p.prior.shape.numel() < p.shape.numel()
        )
        self.do_unsqueeze = do_unsqueeze

    def __enter__(self):
        if not self.do_unsqueeze:
            return

        for p in self.params:
            p.unsqueeze_(self.dim_to_unsqueeze)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise exc_val

        if not self.do_unsqueeze:
            return

        for p in self.params:
            p.squeeze_(self.dim_to_unsqueeze)
