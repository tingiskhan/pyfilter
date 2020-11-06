from typing import Union, Type, Callable, Dict
from torch.distributions import Distribution
import torch
from ..parameter import Parameter
from ...utils import ShapeLike
from ...module import Module


TypeOrCallable = Union[Type[Distribution], Callable[[Dict[str, torch.Tensor]], Distribution]]


class DistributionBuilder(Module):
    def __init__(self, base_dist: TypeOrCallable, **parameters):
        self._base_dist = base_dist
        self._parameters = {k: Parameter(v) if not isinstance(v, Parameter) else v for k, v in parameters.items()}

        try:
            dist = self._base_dist(**self._parameters)
        except Exception as e:
            raise ValueError("Something went wrong when trying to create distribution!") from e

    def get_parameters(self):
        return tuple(v for v in self._parameters.values() if v.trainable)

    def build(self, shape: ShapeLike = None) -> Distribution:
        if shape is None or len(shape) == 0:
            return self._base_dist(**self._parameters)

        new_dict = {k: v.view(shape) if v.trainable else v for k, v in self._parameters.items()}
        return self._base_dist(**new_dict)

    def populate_state_dict(self) -> Dict[str, object]:
        return {
            "_parameters": self._parameters
        }

    __call__ = build