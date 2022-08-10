import torch

from ..state import FilterAlgorithmState
from ...filters import FilterResult, FilterState
from ...utils import normalize, get_ess


class SequentialAlgorithmState(FilterAlgorithmState):
    """
    Base state for sequential particle algorithms.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult):
        """
        Initializes the :class:`SequentialAlgorithmState` class.

        Args:
            weights: the log weights associated with the particle approximation.
            filter_state: the current state of the filter. Somewhat misnamed as we keep track of the entire history of
                the filter, should perhaps be called ``filter_result``.
        """

        super().__init__(filter_state)
        self.w = weights
        self.tensor_tuples.make_deque("ess", get_ess(weights).unsqueeze(0))

    @property
    def ess(self) -> torch.Tensor:
        """
        Returns the ESS.
        """

        return self.tensor_tuples.get_as_tensor("ess")

    def append(self, filter_state: FilterState):
        """
        Updates ``self`` given a new filter state.

        Args:
            filter_state: The latest filter state.
        """

        self.w += filter_state.get_loglikelihood()
        self.tensor_tuples["ess"].append(get_ess(self.w))

    def normalized_weights(self) -> torch.Tensor:
        return normalize(self.w)

    def replicate(self, filter_state):
        return SequentialAlgorithmState(torch.zeros_like(self.w), filter_state)

    def state_dict(self):
        res = super(SequentialAlgorithmState, self).state_dict()
        res["w"] = self.w

        return res

    def load_state_dict(self, state_dict):
        super(SequentialAlgorithmState, self).load_state_dict(state_dict)
        self.w = state_dict["w"]


class SMC2State(SequentialAlgorithmState):
    """
    Custom state class for :class:`pyfilter.inference.sequential.SMC2`, as it requires keeping a history of the parsed
    observations.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult):
        """
        Initializes the :class:`SMC2State` class.

        Args:
            weights: see base:
            filter_state: see base.
        """

        super().__init__(weights, filter_state)
        self.tensor_tuples.make_deque("parsed_data", tuple())

    @property
    def parsed_data(self) -> torch.Tensor:
        return self.tensor_tuples.get_as_tensor("parsed_data")

    def append_data(self, y: torch.Tensor):
        self.tensor_tuples["parsed_data"].append(y)
