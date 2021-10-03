from torch.distributions import Distribution
import torch
from typing import Dict


class DistributionBuilderMixin(object):
    """
    Mixin for "modulizing" distributions.
    """

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        res = dict()

        res.update(self._parameters)
        res.update(self._buffers)

        return res

    def build_distribution(self) -> Distribution:
        return self.base_dist(**self.get_parameters())

    def forward(self) -> Distribution:
        return self.build_distribution()

    @property
    def shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.
        """

        return self.build_distribution().event_shape
