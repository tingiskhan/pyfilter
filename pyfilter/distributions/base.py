from torch.distributions import Distribution, Independent
import torch
from typing import Dict
from .typing import DistributionOrBuilder
from abc import ABC


class DistributionBuilder(torch.nn.Module, ABC):
    """
    Base class for "modulizing" distributions, i.e. representing a ``torch.distributions.Distribution`` object as a
    ``torch.nn.Module``.
    """

    def __init__(self, base_dist: DistributionOrBuilder, reinterpreted_batch_ndims=None):
        """
        Initializes the ``DistributionBuilder`` class.

        Args:
            reinterpreted_batch_ndims: See ``torch.distributions.Independent``.
        """
        
        super(DistributionBuilder, self).__init__()
        
        self.base_dist = base_dist
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def build_distribution(self) -> Distribution:
        """
        Constructs the distribution.
        """

        return self.__call__()

    def forward(self):
        dist = self.base_dist(**self._get_parameters())

        if self._reinterpreted_batch_ndims is None:
            return dist

        return Independent(dist, self._reinterpreted_batch_ndims)

    def _get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Gets the parameters to initialize the distribution with, overridden by derived classes.
        """

        raise NotImplementedError()

    @property
    def shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.
        """

        return self.build_distribution().event_shape
