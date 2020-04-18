import torch


class BaseApproximation(object):
    def __init__(self):
        """
        Base class for constructing variational approximations.
        """
        self._dist = None

    def initialize(self, data, ndim):
        """
        Initializes the approximation.
        :param data: The data to use
        :type data: torch.Tensor
        :param ndim: The dimension of the latent state
        :type ndim: int
        :return: Self
        :rtype: BaseApproximation
        """

        return self

    def entropy(self):
        """
        Returns the entropy of the variational approximation
        :rtype: torch.Tensor
        """

        return self._dist.entropy()

    def sample(self, num_samples=None):
        """
        Samples from the approximation density
        :param num_samples: The number of samples
        :type num_samples: int
        :rtype: torch.Tensor
        """

        samples = (num_samples,) if isinstance(num_samples, int) else num_samples
        return self._dist.rsample(samples or torch.Size([]))

    def get_parameters(self):
        """
        Returns the parameters to optimize.
        :rtype: iterable[torch.Tensor]
        """

        raise NotImplementedError()
