from torch.distributions import Distribution
import torch
from ....state import FilterAlgorithmState
from .....filters import BaseFilter


class BaseProposal(object):
    def build(self, state: FilterAlgorithmState, filter_: BaseFilter, y: torch.Tensor) -> Distribution:
        raise NotImplementedError()

    def exchange(self, old: Distribution, new: Distribution, indices: torch.Tensor) -> None:
        raise NotImplementedError()
