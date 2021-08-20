from typing import Union
from torch.nn import Module
from .prior import Prior
from ..mixins import AllowPriorMixin, DistributionBuilderMixin
from ..mixins.register_parameter_prior import RegisterParameterAndPriorMixin
from .typing import DistributionOrBuilder, HyperParameters


class DistributionWrapper(DistributionBuilderMixin, AllowPriorMixin, RegisterParameterAndPriorMixin, Module):
    """
    Implements a wrapper around pytorch distributions in order to enable moving distributions between devices.
    """

    def __init__(self, base_dist: DistributionOrBuilder, **parameters: Union[HyperParameters, Prior]):
        super().__init__()

        self.base_dist = base_dist
        parameters["validate_args"] = parameters.pop("validate_args", False)

        for k, v in parameters.items():
            self._register_parameter_or_prior(k, v)
