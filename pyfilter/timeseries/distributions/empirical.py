import torch
from torch.distributions import Distribution


class Empirical(Distribution):
    def __init__(self, samples: torch.Tensor):
        """
        Helper class for timeseries without an analytical expression.
        :param samples: The sample
        """
        super().__init__()
        self.loc = self._samples = samples
        self.scale = torch.zeros_like(samples)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape != self._samples.shape and sample_shape != torch.Size():
            raise ValueError("Current implementation only allows passing an empty size!")

        return self._samples