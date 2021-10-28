from torch.distributions import Distribution
import torch
from typing import Type, Callable, Sequence, Union
from numbers import Number

HyperParameter = Union[torch.Tensor, Number]
DistributionOrBuilder = Union[Type[Distribution], Callable[[Sequence[HyperParameter]], Distribution]]
