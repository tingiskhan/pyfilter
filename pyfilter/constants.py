import torch
from math import sqrt


INFTY = float("inf")
EPS = sqrt(torch.finfo(torch.float32).eps)
