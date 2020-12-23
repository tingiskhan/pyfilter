import torch
from typing import List
from ..state import AlgorithmState
from ...filters import FilterResult
from ...utils import normalize, TensorTuple


class FilteringAlgorithmState(AlgorithmState):
    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess: List[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("w", weights)
        self.filter_state = filter_state
        self.ess = ess or list()

    def normalized_weights(self):
        return normalize(self.w)

    def get_ess(self) -> torch.Tensor:
        return torch.stack(self.ess, 0)

    def append_ess(self, ess: torch.Tensor):
        self.ess.append(ess)


class SMC2State(FilteringAlgorithmState):
    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess=None):
        super().__init__(weights, filter_state, ess)
        self.parsed_data = TensorTuple()

    def append_data(self, y: torch.Tensor):
        self.parsed_data.append(y)
