from ..state import BaseState
from ..filters import FilterResult


class AlgorithmState(BaseState):
    """
    Base state class for algorithms.
    """


class FilterAlgorithmState(AlgorithmState):
    """
    Base state class for filter based algorithms.
    """

    def __init__(self, filter_state: FilterResult):
        """
        Initializes the ``FilterAlgorithmState`` class.

        Args:
             filter_state: The initial ``pyfilter.filters.FilterResult``.
        """

        super().__init__()
        self.filter_state = filter_state

    def replicate(self, filter_state: FilterResult) -> "FilterAlgorithmState":
        """
        Creates a replica (not copy) of the instance with given ``filter_state``.

        Args:
            filter_state: The filter state to use when creating the replica.
        """

        return FilterAlgorithmState(filter_state)
