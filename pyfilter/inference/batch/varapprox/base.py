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
        """
        Initializes the approximation.
        :param data: The data to use
        :param ndim: The dimension of the latent state
        :return: Self
        """

        return self

    def dist(self) -> Distribution:
        """
        Returns the distribution.
        """

        raise NotImplementedError()

    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of the variational approximation
        """

        return self.dist().entropy()

    def sample(self, num_samples: int = None) -> torch.Tensor:
        """
        Samples from the approximation density
        :param num_samples: The number of samples
        """

        samples = (num_samples,) if isinstance(num_samples, int) else num_samples
        return self.dist().rsample(samples or torch.Size([]))

    def get_parameters(self) -> Iterable[torch.Tensor]:
        """
        Returns the parameters to optimize.
        """

        raise NotImplementedError()
