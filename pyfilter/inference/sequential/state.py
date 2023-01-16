import torch

from ...filters import FilterResult, Correction
from ...utils import get_ess, normalize
from ..state import FilterAlgorithmState


class SequentialAlgorithmState(FilterAlgorithmState):
    """
    Base state for sequential particle algorithms.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult):
        """
        Internal initializer for :class:`SequentialAlgorithmState`.

        Args:
            weights (torch.Tensor): initial log weights associated with the particle approximation.
            filter_state (FilterResult): current state of the filter.
        """

        super().__init__(filter_state)
        self.w = weights
        self.tensor_tuples.make_deque("ess", get_ess(weights).unsqueeze(0))
        self.current_iteration: int = 0

    @property
    def ess(self) -> torch.Tensor:
        """
        Returns the ESS.
        """

        return self.tensor_tuples.get_as_tensor("ess")

    def append(self, filter_state: Correction):
        """
        Updates ``self`` given a new filter state.

        Args:
            filter_state (Correction): latest filter state.
        """

        self.w += filter_state.get_loglikelihood()
        self.tensor_tuples["ess"].append(get_ess(self.w))

    def bump_iteration(self):
        """
        Bumps current iteration by 1.
        """

        self.current_iteration += 1

    def normalized_weights(self) -> torch.Tensor:
        return normalize(self.w)

    def replicate(self, filter_state):
        return SequentialAlgorithmState(torch.zeros_like(self.w), filter_state)

    def state_dict(self):
        res = super(SequentialAlgorithmState, self).state_dict()
        res["w"] = self.w
        res["current_iteration"] = self.current_iteration

        return res

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.w = state_dict["w"]
        self.current_iteration = state_dict["current_iteration"]


class SMC2State(SequentialAlgorithmState):
    """
    Custom state class for :class:`pyfilter.inference.sequential.SMC2`, as it requires keeping a history of the parsed
    observations.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult):
        """
        Internal initializer for :class:`SMC2State`.

        Args:
            weights: see :class:`SequentialAlgorithmState`.
            filter_state: see :class:`SequentialAlgorithmState`.
        """

        super().__init__(weights, filter_state)
        self.tensor_tuples.make_deque("parsed_data", tuple())

    @property
    def parsed_data(self) -> torch.Tensor:
        return self.tensor_tuples.get_as_tensor("parsed_data")

    def append_data(self, y: torch.Tensor):
        self.tensor_tuples["parsed_data"].append(y)
