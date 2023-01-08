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
        Internal initializer for :class:`BaseProposal`.
        """

    def build(
        self, context: InferenceContext, state: FilterAlgorithmState, filter_: BaseFilter, y: torch.Tensor
    ) -> Distribution:
        r"""
        Method to be overridden by derived classes. Generates a proposal kernel from which to sample a candidate sample :math:`\theta^*`.

        Args:
            context (InferenceContext): context to use.
            state (FilterAlgorithmState): see :meth:`pyfilter.inference.batch.mcmc.utils.run_pmmh`.
            filter_ (BaseFilter): see :meth:`pyfilter.inference.batch.mcmc.utils.run_pmmh`.
            y (torch.Tensor): see :meth:`pyfilter.inference.batch.mcmc.utils.run_pmmh`.
        """

        raise NotImplementedError()

    def exchange(self, latest: Distribution, candidate: Distribution, mask: torch.Tensor) -> None:
        """
        Method to be overridden by derived classes. Given the latest and candidate kernel, together with the accepted
        mask, do an exchange of distributional parameters between ``latest`` and ``candidate``, where ``latest``
        should be given the parameters of ``candidate``.

        Args:
            latest (Distribution): kernel whose parameters are to be exchange with ``candidate``.
            candidate (Distribution): candidate kernel.
            mask (torch.Tensor): mask of parameters of ``candidate`` to exchange with ``latest``.
        """

        raise NotImplementedError()
