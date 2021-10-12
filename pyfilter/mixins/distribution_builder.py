from torch.distributions import Distribution
import torch
from typing import Dict


class DistributionBuilderMixin(object):
    """
    Mixin for "modulizing" distributions, i.e. representing a ``torch.distributions.Distribution`` object as a
    ``torch.nn.Module``.
    """

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns the a parameters as a dictionary.
        """

        res = dict()

        res.update(self._parameters)
        res.update(self._buffers)

        return res

    def build_distribution(self) -> Distribution:
        """
        Constructs the distribution.
        """

        return self.base_dist(**self.get_parameters())

    def forward(self) -> Distribution:
        return self.build_distribution()

    @property
    def shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.
        """

        return self.build_distribution().event_shape
