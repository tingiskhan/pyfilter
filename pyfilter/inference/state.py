from ..state import StateWithTensorTuples
from ..filters import FilterResult


class AlgorithmState(StateWithTensorTuples):
    """
    Base class for algorithm states.
    """


class FilterAlgorithmState(AlgorithmState):
    """
    Base class for filter based algorithms.
    """

    def __init__(self, filter_state: FilterResult):
        super().__init__()
        self.filter_state = filter_state

    def copy(self, filter_state: FilterResult):
        return FilterAlgorithmState(filter_state)
