from torch.distributions import Distribution
import torch
from typing import Type, Dict, Union, Callable
from ..prior_module import PriorModule
from .mixin import BuilderMixin
from .prior import Prior


DistributionType = Union[Type[Distribution], Callable[[Dict], Distribution]]


class DistributionWrapper(BuilderMixin, PriorModule):
    def __init__(self, base_dist: DistributionType, **parameters):
        super().__init__()

        self.base_dist = base_dist

        for k, v in parameters.items():
            if isinstance(v, Prior):
                self.register_prior(k, v)
            else:
                self.register_buffer(k, v if isinstance(v, torch.Tensor) else torch.tensor(v))
