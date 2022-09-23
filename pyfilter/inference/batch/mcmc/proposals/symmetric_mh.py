import torch
from .base import BaseProposal
from ....utils import construct_mvn


# TODO: Should perhaps create a quasi-specific class for this...?
class SymmetricMH(BaseProposal):
    """
    Builds a symmetric proposal as defined in the original `SMC2` paper.
    """

    def build(self, context, state, filter_, y):
        values = context.stack_parameters(constrained=False)
        weights = state.normalized_weights()

        return construct_mvn(values, weights, scale=1.1, quasi=context.is_quasi)  # Same scale in in num_particles

    def exchange(self, latest, candidate, mask: torch.Tensor) -> None:
        return
