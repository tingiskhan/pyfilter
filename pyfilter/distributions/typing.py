from torch.distributions import Distribution
import torch
from typing import Type, Callable, Sequence, Union
from .prior import Prior


DistributionOrBuilder = Union[
    Type[Distribution],
    Callable[[Type[Distribution], Sequence[Union[torch.Tensor, float]]], Distribution]
]

Parameters = Union[torch.Tensor, float, int, Prior]