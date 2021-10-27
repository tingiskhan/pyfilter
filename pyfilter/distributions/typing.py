from torch.distributions import Distribution
import torch
from typing import Type, Callable, Sequence, Union

HyperParameter = Union[torch.Tensor, float, int]
DistributionOrBuilder = Union[Type[Distribution], Callable[[Sequence[HyperParameter]], Distribution]]
