from torch.distributions import Distribution
import numpy as np
from typing import Union, Tuple
import torch
from .distributions import Prior
from .timeseries.timeseriesstate import TimeseriesState


StateLike = Union[Distribution, TimeseriesState, Prior]
ArrayType = Union[float, int, np.ndarray, StateLike]
ShapeLike = Union[int, Tuple[int, ...], torch.Size]