from torch.distributions import Distribution
import torch
from .utils import construct_mvn


class SampleMVN(Distribution):
    """
    Sample based MVN.
    """

    arg_constraints = {}

    def __init__(self, samples: torch.Tensor, weights: torch.Tensor, scale: float = 1.1, **kwargs):
        super().__init__(validate_args=kwargs.pop("validate_args", False), **kwargs)

        self.samples = samples
        self.weights = weights
        self.scale = scale

    def rsample(self, sample_shape=torch.Size()):
        return construct_mvn(self.samples, self.weights, self.scale).rsample(sample_shape)

    def log_prob(self, value):
        return construct_mvn(self.samples, self.weights, self.scale).log_prob(value)
