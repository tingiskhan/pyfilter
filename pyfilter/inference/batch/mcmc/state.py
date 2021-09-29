import torch
from ...state import FilterAlgorithmState
from ....filters import FilterResult
from ....utils import TensorTuple


class PMMHState(FilterAlgorithmState):
    """
    State object for PMMH algorithm.
    """

    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        """
        Initializes the ``PMMHState`` class.

        Args:
            initial_sample: The initial sample of the Markov chain.
            filter_result: The filter result object.
        """

        super().__init__(filter_result)
        self.samples = TensorTuple(initial_sample)

    def update_chain(self, sample: torch.Tensor):
        """
        Updates the Markov chain with the newly accepted candidate.

        Args:
            sample: The next accepted sample of the chain.
        """

        self.samples.append(sample)
