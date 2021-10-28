import numpy as np
from typing import Union, Tuple
import torch
from numbers import Number
from .distributions import Prior


ArrayType = Union[Number, np.ndarray, Prior, torch.Tensor]
ShapeLike = Union[int, Tuple[int, ...], torch.Size]
