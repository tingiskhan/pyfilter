from torch.distributions import Distribution
import torch
from torch.nn import Module
from typing import Type, Dict, Union, Callable


DistributionType = Union[Type[Distribution], Callable[[Dict], Distribution]]


class DistributionWrapper(Module):
    def __init__(self, base_dist: DistributionType, **parameters):
        super().__init__()

        self.base_dist = base_dist

        for k, v in parameters.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)

            self.register_buffer(k, v)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        res = dict()

        res.update(self._parameters)
        res.update(self._buffers)

        return res

    def build_distribution(self) -> Distribution:
        return self.base_dist(**self.get_parameters())

    def forward(self):
        return self.build_distribution()
