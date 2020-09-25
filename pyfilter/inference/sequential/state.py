from ..state import AlgorithmState
from ...filters import BaseState
from torch import Tensor


class FilteringAlgorithmState(AlgorithmState):
    def __init__(self, weights: Tensor, filter_state: BaseState):
        self.w = weights
        self.filter_state = filter_state
