from typing import Union, Type, Callable, Dict
from torch.distributions import Distribution
import torch
from .timeseries.parameter import Parameter, ShapeLike


TypeOrCallable = Union[Type[Distribution], Callable[[Dict[str, torch.Tensor]], Distribution]]


class DistributionBuilder(object):
    def __init__(self, base_dist: TypeOrCallable, **parameters):
        self._base_dist = base_dist
        self._parameters = parameters

    def get_parameters(self):
        return tuple(v for v in self._parameters.values() if isinstance(v, Parameter))

    def build(self, shape: ShapeLike = None) -> Distribution:
        if shape is None:
            return self._base_dist(**self._parameters)

        new_dict = {k: v.view(shape) if isinstance(v, torch.Tensor) else v for k, v in self._parameters.items()}
        return self._base_dist(**new_dict)