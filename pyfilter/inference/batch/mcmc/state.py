import torch
from ...state import FilterAlgorithmState
from ....filters import FilterResult
from ....container import add_right
from ....timeseries import StateSpaceModel


class PMMHResult(FilterAlgorithmState):
    """
    Result object for PMMH algorithm.
    """

    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        """
        Initializes the ``PMMHResult`` class.

        Args:
            initial_sample: The initial sample of the Markov chain.
            filter_result: The filter result object.
        """

        super().__init__(filter_result)
        self.tensor_tuples["samples"] = initial_sample

    @property
    def samples(self) -> torch.Tensor:
        return torch.stack(self.tensor_tuples["samples"], dim=0)

    def update_chain(self, sample: torch.Tensor):
        """
        Updates the Markov chain with the newly accepted candidate.

        Args:
            sample: The next accepted sample of the chain.
        """

        self.tensor_tuples["samples"] += (sample,)

    def update_parameters_from_chain(self, model: StateSpaceModel, burn_in: int, constrained=True):
        """
        Sets the parameters of ``model`` from chain indexed from ``burn_in`` and forward.

        Args:
            model: The model to set the parameters for.
            burn_in: The number of samples from the chain to discard.
            constrained: Whether parameters are constrained.
        """

        samples = self.samples[burn_in:]
        samples = samples.flatten(end_dim=-2)

        model.sample_params(torch.Size([samples.shape[0], 1]))
        model.update_parameters_from_tensor(samples, constrained=constrained)
