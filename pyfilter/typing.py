from torch.distributions import Distribution
import numpy as np
from typing import Union, Tuple, Callable
import torch
from .distributions import Prior


StateLike = Union[Distribution, "TimeseriesState", Prior]
ArrayType = Union[float, int, np.ndarray, StateLike]
ShapeLike = Union[int, Tuple[int, ...], torch.Size]
MeanOrScaleFun = Callable[[StateLike, Tuple[torch.Tensor, ...]], torch.Tensor]
DiffusionFunction = Callable[["TimeseriesState", float, Tuple[torch.Tensor, ...]], Distribution]
