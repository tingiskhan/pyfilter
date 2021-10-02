from torch.distributions import Distribution
import torch
from typing import Type, Callable, Sequence, Union

HyperParameters = Union[torch.Tensor, float, int]
DistributionOrBuilder = Union[Type[Distribution], Callable[[Sequence[HyperParameters]], Distribution]]
