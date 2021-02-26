import torch
from typing import Union
from ..prior_module import PriorModule
from .mixin import BuilderMixin
from .prior import Prior
from .typing import DistributionOrBuilder, Parameters


class DistributionWrapper(BuilderMixin, PriorModule):
    def __init__(self, base_dist: DistributionOrBuilder, **parameters: Union[Parameters, Prior]):
        super().__init__()

        self.base_dist = base_dist

        for k, v in parameters.items():
            if isinstance(v, Prior):
                self.register_prior(k, v)
            else:
                self.register_buffer(k, v if isinstance(v, torch.Tensor) else torch.tensor(v))
