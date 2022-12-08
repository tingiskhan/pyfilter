from abc import ABC

import torch
from torch.distributions import Distribution

from .....filters import BaseFilter
from ....context import InferenceContext
from ....state import FilterAlgorithmState


class BaseProposal(ABC):
    """
    Abstract base class for proposal objects used for generating candidate num_samples in PMMH.
    """

    def __init__(self):
        """
        Initializes the :class:`BaseProposal` class.
        """

    def build(
        self, context: InferenceContext, state: FilterAlgorithmState, filter_: BaseFilter, y: torch.Tensor
    ) -> Distribution:
        r"""
        Method to be overridden by derived classes. Given the latest state, filter and dataset, generate a proposal
        kernel from which to sample a candidate sample :math:`\theta^*`.

        Args:
            context: the context to use.
            state: see :meth:`pyfilter.inference.batch.mcmc.utils.run_pmmh`.
            filter_: see :meth:`pyfilter.inference.batch.mcmc.utils.run_pmmh`.
            y: see :meth:`pyfilter.inference.batch.mcmc.utils.run_pmmh`.
        """

        raise NotImplementedError()

    def exchange(self, latest: Distribution, candidate: Distribution, mask: torch.Tensor) -> None:
        """
        Method to be overridden by derived classes. Given the latest and candidate kernel, together with the accepted
        mask, do an exchange of distributional parameters between ``latest`` and ``candidate``, where ``latest``
        should be given the parameters of ``candidate``.

        Args:
            latest: the latest kernel, the distribution that is supposed to receive distributional parameters from
                ``candidate``.
            candidate: The candidate kernel.
            mask: the accepted mask of ``candidate``.

        """

        raise NotImplementedError()
