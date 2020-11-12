from .model import StateSpaceModel
from .observable import AffineObservations
import torch
from torch import distributions as dists
from typing import Union


def f_0d(x, a, scale):
    return a * x


def f_1d(x, a, scale):
    return f_2d(x, a.unsqueeze(-2), scale)[..., 0]


def f_2d(x, a, scale):
    return torch.matmul(a, x.unsqueeze(-1))[..., 0]


def g(x, a, *scale):
    if len(scale) == 1:
        return scale[-1]

    return scale


def _get_shape(a):
    is_1d = False
    if isinstance(a, dists.Distribution):
        dim = a.event_shape
        is_1d = len(a.event_shape) == 1
    elif isinstance(a, float) or a.dim() < 2:
        dim = torch.Size([])
        is_1d = (torch.tensor(a) if isinstance(a, float) else a).dim() <= 1
    else:
        dim = a.shape[:1]

    return dim, is_1d


class LinearObservations(StateSpaceModel):
    def __init__(
        self,
        hidden,
        a: Union[torch.Tensor, float, dists.Distribution],
        scale: Union[torch.Tensor, float, dists.Distribution],
        base_dist,
    ):
        """
        Defines a class of observation dynamics where the observed variable is a linear combination of the states.
        :param hidden: The hidden dynamics
        :param a: The A-matrix
        :param scale: The variance of the observations
        :param base_dist: The base distribution
        """

        # ===== Convoluted way to decide number of dimensions ===== #
        dim, is_1d = _get_shape(a)

        # ===== Assert distributions make sense ===== #
        if base_dist.event_shape != dim:
            raise ValueError("The distribution is not of correct shape!")

        # ===== Determine propagator function ===== #
        if not is_1d:
            f = f_2d
        elif is_1d and hidden.ndim > 0:
            f = f_1d
        else:
            f = f_0d

        observable = AffineObservations((f, g), (a, scale), base_dist)

        super().__init__(hidden, observable)


class LinearGaussianObservations(LinearObservations):
    def __init__(self, hidden, a=1.0, scale=1.0):
        """
        Implements a State Space model that's linear in the observation equation but has arbitrary dynamics in the
        state process.
        :param hidden: The hidden dynamics
        :param a: The A-matrix
        :param scale: The variance of the observations
        """

        # ===== Convoluted way to decide number of dimensions ===== #
        dim, is_1d = _get_shape(a)

        # ====== Define distributions ===== #
        n = dists.Normal(0.0, 1.0) if is_1d else dists.Independent(dists.Normal(torch.zeros(dim), torch.ones(dim)), 1)

        if not isinstance(scale, (torch.Tensor, float, dists.Distribution)):
            raise ValueError(f"`scale` parameter must be numeric type!")

        super().__init__(hidden, a, scale, n)
