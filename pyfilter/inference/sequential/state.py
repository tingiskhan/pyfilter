from ..state import AlgorithmState
from ...filters import FilterResult
from torch import Tensor


class FilteringAlgorithmState(AlgorithmState):
    def __init__(self, weights: Tensor, filter_state: FilterResult):
        self.w = weights
        self.filter_state = filter_state
