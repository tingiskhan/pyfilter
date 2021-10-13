import torch
from typing import List
from ..state import FilterAlgorithmState
from ...filters import FilterResult
from ...utils import normalize
from ...container import TensorTuple


class SequentialAlgorithmState(FilterAlgorithmState):
    """
    Base state for sequential particle algorithms.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess: List[torch.Tensor] = None):
        """
        Initializes the ``SequentialAlgorithmState`` class.

        Args:
            weights: The log weights associated with the particle approximation.
            filter_state: The current state of the filter. Somewhat misnamed as we keep track of the entire history of
                the filter, should perhaps be called ``filter_result``.
            ess: Optional parameter, only used when re-initializing a state object.
        """

        super().__init__(filter_state)
        self.register_buffer("w", weights)
        self.ess = ess or list()

    def normalized_weights(self) -> torch.Tensor:
        return normalize(self.w)

    def get_ess(self) -> torch.Tensor:
        return torch.stack(self.ess, 0)

    def append_ess(self, ess: torch.Tensor):
        self.ess.append(ess)

    def replicate(self, filter_state):
        return SequentialAlgorithmState(torch.zeros_like(self.w), filter_state)


class SMC2State(SequentialAlgorithmState):
    """
    Custom state class for ``SMC2``, as it requires keeping a history of the parsed observations.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess=None, parsed_data: TensorTuple = None):
        """
        Initializes the ``SMC2State`` class.

        Args:
            weights: See base:
            filter_state: See base.
            ess: See base.
            parsed_data: The collection of observations that have been parsed by the algorithm.
        """

        super().__init__(weights, filter_state, ess)
        self.tensor_tuples["parsed_data"] = parsed_data or TensorTuple()

    @property
    def parsed_data(self) -> TensorTuple:
        return self.tensor_tuples["parsed_data"]

    def append_data(self, y: torch.Tensor):
        self.parsed_data.append(y)
