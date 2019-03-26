

class BaseApproximation(object):
    """
    A base class for constructing variational approximations of the latent states.
    """

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

        raise NotImplementedError()

    def sample(self, num_samples):
        """
        Samples from the approximation density
        :param num_samples: The number of samples
        :type num_samples: int
        :rtype: torch.Tensor
        """

        raise NotImplementedError()

    def get_parameters(self):
        """
        Returns the parameters to optimize.
        :rtype: iterable[torch.Tensor]
        """

        raise NotImplementedError()
