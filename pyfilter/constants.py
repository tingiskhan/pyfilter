import torch
from math import sqrt


INFTY = float("inf")
EPS = sqrt(torch.finfo(torch.get_default_dtype()).eps)
EPS2 = EPS ** 2.0
