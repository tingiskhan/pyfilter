from torch.distributions import Distribution
import torch
from .utils import construct_mvn


class SampleMVN(Distribution):
    """
    Sample based MVN.
    """

    def __init__(self, samples: torch.Tensor, weights: torch.Tensor, scale: float = 1.1):
        super().__init__()

        self.samples = samples
        self.weights = weights
        self.scale = scale

    def rsample(self, sample_shape=torch.Size()):
        return construct_mvn(self.samples, self.weights, self.scale).rsample(sample_shape)

    def log_prob(self, value):
        return construct_mvn(self.samples, self.weights, self.scale).log_prob(value)
