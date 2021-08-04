import torch
from ....utils import params_to_tensor
from .base import BaseProposal
from .....distributions.utils import construct_mvn


# TODO: Inherit from other subclass?
class SymmetricMH(BaseProposal):
    """
    Builds a symmetric proposal, used in SMC2.
    """

    def build(self, state, filter_, y):
        values = params_to_tensor(filter_.ssm, constrained=False)
        weights = state.normalized_weights()

        return construct_mvn(values, weights, scale=1.1)  # Same scale in in particles

    def exchange(self, old, new, indices: torch.Tensor) -> None:
        return
