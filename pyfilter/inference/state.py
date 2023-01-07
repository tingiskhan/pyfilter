from copy import deepcopy

from ..filters import FilterResult
from ..state import BaseResult


class AlgorithmState(BaseResult):
    """
    Base state class for algorithms.
    """


class FilterAlgorithmState(AlgorithmState):
    """
    Base state class for filter based algorithms.
    """

    def __init__(self, filter_state: FilterResult):
        """
        Internal initializer for :class:`FilterAlgorithmState`.

        Args:
             filter_state: the initial :class:`pyfilter.filters.FilterResult`.
        """

        super().__init__()
        self.filter_state = filter_state

    def replicate(self, filter_state: FilterResult) -> "FilterAlgorithmState":
        """
        Creates a replica (not copy) of the instance with given ``filter_state``.

        Args:
            filter_state: the filter state to use when creating the replica.
        """

        return FilterAlgorithmState(filter_state)

    def state_dict(self):
        res = super(FilterAlgorithmState, self).state_dict()
        res["filter_state"] = self.filter_state.state_dict()

        return res

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.filter_state.load_state_dict(state_dict["filter_state"])

    def __repr__(self):
        return f"{self.__class__.__name__}(ll: {self.filter_state.loglikelihood})"

    def copy(self) -> "FilterAlgorithmState":
        r"""
        Creates a copy of self.
        """

        # NB: This is untested and might not be optimal tbh
        return deepcopy(self)
