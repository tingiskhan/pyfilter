from torch.distributions import Distribution
import torch


class DistributionBuilderMixin(object):
    """
    Mixin for "modulizing" distributions, i.e. representing a ``torch.distributions.Distribution`` object as a
    ``torch.nn.Module``.
    """

    def build_distribution(self) -> Distribution:
        """
        Constructs the distribution.
        """

        return self.__call__()

    @property
    def shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.
        """

        return self.build_distribution().event_shape
