from .base import BaseApproximation
import torch
from torch.distributions import Independent, Normal


class MeanField(BaseApproximation):
    def __init__(self):
        """
        Implements a mean field approximation of the state space
        """

        self._mean = None
        self._logstd = None

        self._sampledist = None     # type: Independent

    def entropy(self):
        return Independent(Normal(self._mean, self._logstd.exp()), 2).entropy()

    def initialize(self, data, ndim):
        self._mean = torch.zeros((data.shape[0] + 1, ndim), requires_grad=True)
        self._logstd = torch.zeros_like(self._mean, requires_grad=True)

        # ===== Start optimization ===== #
        self._sampledist = Independent(Normal(torch.zeros_like(self._mean), torch.ones_like(self._logstd)), 2)

        return self

    def get_parameters(self):
        return [self._mean, self._logstd]

    def sample(self, num_samples):
        samples = (num_samples,) if isinstance(num_samples, int) else num_samples
        return self._mean + self._logstd.exp() * self._sampledist.sample(samples)


# TODO: Only supports 1D parameters currently
class ParameterApproximation(MeanField):
    def initialize(self, parameters, *args):
        self._mean = torch.zeros(len(parameters))
        self._logstd = torch.zeros_like(self._mean)

        for i, p in enumerate(parameters):
            self._mean[i] = p.bijection.inv(p.distr.mean)
            self._logstd[i] = p.distr.stddev.log()

        self._mean.requires_grad_(True)
        self._logstd.requires_grad_(True)

        self._sampledist = Independent(Normal(torch.zeros_like(self._mean), torch.ones_like(self._logstd)), 1)

        return self

    def entropy(self):
        return Independent(Normal(self._mean, self._logstd.exp()), 1).entropy()