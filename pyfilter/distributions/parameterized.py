import torch
from typing import Union
from torch.nn import Module, Parameter
from .prior_mixin import PriorMixin
from .mixin import BuilderMixin
from .prior import Prior
from .typing import DistributionOrBuilder, Parameters


class DistributionWrapper(BuilderMixin, PriorMixin, Module):
    """
    Implements a wrapper around pytorch distributions in order to enable moving distributions between devices.
    """

    def __init__(self, base_dist: DistributionOrBuilder, **parameters: Union[Parameters, Prior]):
        super().__init__()

        self.base_dist = base_dist
        parameters["validate_args"] = parameters.pop("validate_args", False)

        for k, v in parameters.items():
            if isinstance(v, Prior):
                self.register_prior(k, v)
            elif isinstance(v, Parameter):
                self.register_parameter(k, v)
            else:
                self.register_buffer(k, v if isinstance(v, torch.Tensor) else torch.tensor(v))
