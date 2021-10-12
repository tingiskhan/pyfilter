from torch.distributions import Distribution
import torch
from abc import ABC
from ....state import FilterAlgorithmState
from .....filters import BaseFilter


class BaseProposal(ABC):
    """
    Abstract base class for proposal objects used for generating candidate samples in PMMH.
    """

    def build(self, state: FilterAlgorithmState, filter_: BaseFilter, y: torch.Tensor) -> Distribution:
        """
        Method to be overridden by derived classes. Given the latest state, filter and dataset, generate a proposal
        kernel from which to sample a candidate sample :math:`\\theta^*`.

        Args:
            state: See ``pyfilter.inference.batch.mcmc.utils.run_pmmh``.
            filter_: See ``pyfilter.inference.batch.mcmc.utils.run_pmmh``.
            y: See ``pyfilter.inference.batch.mcmc.utils.run_pmmh``.
        """

        raise NotImplementedError()

    def exchange(self, latest: Distribution, candidate: Distribution, indices: torch.Tensor) -> None:
        """
        Method to be overridden by dervied classes. Given the latest and candidate kernel, together with the accepted
        indices, do an exchange of distributional parameters between ``latest`` and ``candidate``, where ``latest``
        should be given the parameters of ``candidate``.

        Args:
            latest: The latest kernel, the distribution that is supposed to receive distributional parameters from
                ``candidate``.
            candidate: The candidate kernel.
            indices: The accepted indices of ``candidate``.

        Returns:
            Nothing, ``latest`` is update in place.
        """

        raise NotImplementedError()
