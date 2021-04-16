import torch
from torch.distributions import Distribution


class Empirical(Distribution):
    """
    Helper class for timeseries without an analytical expression.
    """

    arg_constraints = {}

    def __init__(self, samples: torch.Tensor):
        super().__init__()
        self._samples = samples

    def sample(self, sample_shape=torch.Size()):
        if sample_shape != self._samples.shape and sample_shape != torch.Size():
            raise ValueError("Current implementation only allows passing an empty size!")

        return self._samples
