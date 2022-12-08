import torch
from .base import BaseProposal
from ....utils import construct_mvn
from ....context import QuasiInferenceContext


class SymmetricMH(BaseProposal):
    """
    Builds a symmetric proposal as defined in the original `SMC2` paper.
    """

    def build(self, context, state, filter_, y):
        values = context.stack_parameters(constrained=False)
        weights = state.normalized_weights()

        # TODO: Fix this by passing context reference instead... Much safer
        return construct_mvn(
            values,
            weights,
            scale=1.1,
            quasi_key=context.quasi_key if isinstance(context, QuasiInferenceContext) else None,
        )

    def exchange(self, latest, candidate, mask: torch.Tensor) -> None:
        return
