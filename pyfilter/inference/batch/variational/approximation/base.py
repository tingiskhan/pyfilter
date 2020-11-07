import torch
from typing import Iterable
from torch.distributions import Distribution


class BaseApproximation(object):
    def __init__(self):
        """
        Base class for constructing variational approximations.
        """
        self._dist = None

    def initialize(self, data: torch.Tensor, ndim: int):
        return self

    def dist(self) -> Distribution:
        raise NotImplementedError()

    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of the variational approximation
        """

        return self.dist().entropy()

    def sample(self, num_samples: int = None) -> torch.Tensor:
        samples = (num_samples,) if isinstance(num_samples, int) else num_samples
        return self.dist().rsample(samples or torch.Size([]))

    def get_parameters(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError()
