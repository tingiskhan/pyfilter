import torch
from .base import BaseProposal
from .....distributions.utils import construct_mvn


class SymmetricMH(BaseProposal):
    """
    Builds a symmetric proposal as defined in the original `SMC2` paper.
    """

    def build(self, state, filter_, y):
        values = filter_.ssm.concat_parameters(constrained=False)
        weights = state.normalized_weights()

        return construct_mvn(values, weights, scale=1.1)  # Same scale in in particles

    def exchange(self, latest, candidate, indices: torch.Tensor) -> None:
        return
