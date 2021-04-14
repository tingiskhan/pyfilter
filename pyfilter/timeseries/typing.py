from torch.distributions import Distribution
from typing import Tuple, Callable
import torch
from .state import NewState


MeanOrScaleFun = Callable[[NewState, Tuple[torch.Tensor, ...]], torch.Tensor]
DiffusionFunction = Callable[[NewState, float, Tuple[torch.Tensor, ...]], Distribution]
