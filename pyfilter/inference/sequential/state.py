import torch
from ..state import FilterAlgorithmState
from ...filters import FilterResult, FilterState
from ...utils import normalize, get_ess


class SequentialAlgorithmState(FilterAlgorithmState):
    """
    Base state for sequential particle algorithms.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess: torch.Tensor = None):
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
        self.tensor_tuples["ess"] = get_ess(weights) if ess is None else tuple(ess)

    @property
    def ess(self) -> torch.Tensor:
        """
        Returns the ESS.
        """

        return torch.stack(self.tensor_tuples["ess"], dim=0)

    def update(self, filter_state: FilterState):
        """
        Updates ``self`` given a new filter state.

        Args:
            filter_state: The latest filter state.
        """

        self.w += filter_state.get_loglikelihood()
        self.filter_state.append(filter_state)

        self.tensor_tuples["ess"] += (get_ess(self.w),)

    def normalized_weights(self) -> torch.Tensor:
        return normalize(self.w)

    def replicate(self, filter_state):
        return SequentialAlgorithmState(torch.zeros_like(self.w), filter_state)


class SMC2State(SequentialAlgorithmState):
    """
    Custom state class for ``SMC2``, as it requires keeping a history of the parsed observations.
    """

    def __init__(self, weights: torch.Tensor, filter_state: FilterResult, ess=None, parsed_data: torch.Tensor = None):
        """
        Initializes the ``SMC2State`` class.

        Args:
            weights: See base:
            filter_state: See base.
            ess: See base.
            parsed_data: The collection of observations that have been parsed by the algorithm.
        """

        super().__init__(weights, filter_state, ess)
        self.tensor_tuples["parsed_data"] = tuple() if parsed_data is None else tuple(parsed_data)

    @property
    def parsed_data(self) -> torch.Tensor:
        return torch.stack(self.tensor_tuples["parsed_data"], dim=0)

    def append_data(self, y: torch.Tensor):
        self.tensor_tuples["parsed_data"] += (y,)
