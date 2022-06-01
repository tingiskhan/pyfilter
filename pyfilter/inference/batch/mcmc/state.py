from typing import Dict

import torch
from stochproc.timeseries import StateSpaceModel
from ...state import FilterAlgorithmState
from ....filters import FilterResult


class PMMHResult(FilterAlgorithmState):
    """
    Result object for PMMH algorithm.
    """

    def __init__(self, initial_sample: Dict[str, torch.Tensor], filter_result: FilterResult, stack_dim=1):
        """
        Initializes the :class:`PMMHResult` class.

        Args:
            initial_sample: the initial sample of the Markov chain.
            filter_result: the filter result object.
            stack_dim: which dimension to stack along.
        """

        super().__init__(filter_result)
        self.dim = stack_dim
        self.samples = {n: v.clone().unsqueeze(self.dim) for n, v in initial_sample.items()}

    def update_chain(self, sample: Dict[str, torch.Tensor]):
        """
        Updates the Markov chain with the newly accepted candidate.

        Args:
            sample: the next accepted sample of the chain.
        """

        for n, p in sample.items():
            sub_sample = self.samples[n]
            self.samples[n] = torch.cat((sub_sample, p.data.clone().unsqueeze(self.dim)), dim=self.dim)

    def update_parameters_from_chain(self, model: StateSpaceModel, burn_in: int, constrained=True):
        """
        Sets the parameters of ``model`` from chain indexed from ``burn_in`` and forward.

        Args:
            model: the model to set the parameters for.
            burn_in: the number of num_samples from the chain to discard.
            constrained: whether parameters are constrained.
        """

        samples = self.samples[burn_in:]
        samples = samples.flatten(end_dim=-2)

        model.sample_params(torch.Size([samples.shape[0], 1]))
        model.update_parameters_from_tensor(samples, constrained=constrained)
