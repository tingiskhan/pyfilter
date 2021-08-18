import torch
from typing import List
from ..state import FilterAlgorithmState
from ...filters import FilterResult
from ...utils import normalize, TensorTuple


class FilteringAlgorithmState(FilterAlgorithmState):
    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess: List[torch.Tensor] = None):
        super().__init__(filter_state)
        self.register_buffer("w", weights)
        self.ess = ess or list()

    def normalized_weights(self):
        return normalize(self.w)

    def get_ess(self) -> torch.Tensor:
        return torch.stack(self.ess, 0)

    def append_ess(self, ess: torch.Tensor):
        self.ess.append(ess)

    def copy(self, filter_state):
        return FilteringAlgorithmState(torch.zeros_like(self.w), filter_state)


class SMC2State(FilteringAlgorithmState):
    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess=None, parsed_data: TensorTuple = None):
        super().__init__(weights, filter_state, ess)
        self.parsed_data = parsed_data or TensorTuple()

        self._register_state_dict_hook(TensorTuple.dump_hook)
        self._register_load_state_dict_pre_hook(lambda *args: TensorTuple.load_hook(self, *args))

    def append_data(self, y: torch.Tensor):
        self.parsed_data.append(y)
