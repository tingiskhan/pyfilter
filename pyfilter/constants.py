from math import sqrt

import torch

INFTY = float("inf")
_info = torch.finfo(torch.get_default_dtype())

EPS = sqrt(_info.eps)
EPS2 = _info.eps

MAX = _info.max
