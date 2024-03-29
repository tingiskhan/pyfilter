from typing import Dict

import torch

from ....filters import FilterResult
from ...state import FilterAlgorithmState


class PMMHResult(FilterAlgorithmState):
    """
    Result object for PMMH algorithm.
    """

    def __init__(self, initial_sample: Dict[str, torch.Tensor], filter_result: FilterResult, stack_dim=1):
        """
        Internal initializer for :class:`PMMHResult`.

        Args:
            initial_sample (Dict[str, torch.Tensor]): initial sample of the Markov chain.
            filter_result (FilterResult): filter result object.
            stack_dim (int, optional): which dimension to stack along. Defaults to 1.
        """

        super().__init__(filter_result)
        self.dim = stack_dim
        self.samples = {n: v.clone().unsqueeze(self.dim) for n, v in initial_sample.items()}

    def update_chain(self, sample: Dict[str, torch.Tensor]):
        """
        Updates the Markov chain with the newly accepted candidate.

        Args:
            sample (Dict[str, torch.Tensor]): next accepted sample of the chain.
        """

        for n, p in sample.items():
            sub_sample = self.samples[n]
            self.samples[n] = torch.cat((sub_sample, p.data.clone().unsqueeze(self.dim)), dim=self.dim)
