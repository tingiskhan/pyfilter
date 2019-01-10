from .model import StateSpaceModel
from .base import BaseModel, Observable
import torch
from torch import distributions as dists


def f(x, a, scale):
    if a.dim() < 1:
        return a * x

    if a.dim() == 1:
        return x @ a

    return torch.matmul(a, x.unsqueeze(-1))[..., 0]


def g(x, a, scale):
    return scale


class LinearGaussianObservations(StateSpaceModel):
    def __init__(self, hidden, a=1., scale=1.):
        """
        Implements a State Space model that's linear in the observation equation but has arbitrary dynamics in the
        state process.
        :param hidden: The hidden dynamics
        :type hidden: BaseModel
        :param a: The A-matrix, must be constant (currently)
        :type a: torch.Tensor|float
        :param scale: The variance of the observations, can be constant or learnable. Currently assumes that all
                      components are independent.
        :type scale: torch.Tensor|dists.Distribution|float
        """

        if isinstance(a, float) or a.dim() < 2:
            noise = dists.Normal(torch.zeros(1), 1)
        else:
            noise = dists.Independent(dists.Normal(torch.zeros(a.dim()), torch.ones(a.dim())), 1)

        observable = Observable((f, g), (a, scale), noise)

        super().__init__(hidden, observable)