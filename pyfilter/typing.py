import numpy as np
from typing import Union, Tuple
import torch
from .distributions import Prior


ArrayType = Union[float, int, np.ndarray, Prior, torch.Tensor]
ShapeLike = Union[int, Tuple[int, ...], torch.Size]
