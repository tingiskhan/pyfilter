from torch.distributions import Distribution
import torch
from typing import Dict


class BuilderMixin(object):
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        res = dict()

        res.update(self._parameters)
        res.update(self._buffers)

        return res

    def build_distribution(self) -> Distribution:
        return self.base_dist(**self.get_parameters())

    def forward(self) -> Distribution:
        return self.build_distribution()