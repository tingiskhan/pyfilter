import torch
from ....utils import params_to_tensor
from .base import BaseProposal
from .....distributions import SampleMVN


class SymmetricMH(BaseProposal):
    def build(self, state, filter_, y):
        values = params_to_tensor(filter_.ssm, constrained=False)
        weights = state.normalized_weights()

        return SampleMVN(values, weights, scale=1.1)  # Same scale in in particles

    def exchange(self, old: SampleMVN, new: SampleMVN, indices: torch.Tensor) -> None:
        old.samples[indices] = new.samples[indices]
        old.weights = torch.ones_like(new.weights) / new.weights.shape[0]
