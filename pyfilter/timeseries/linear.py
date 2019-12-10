from .model import StateSpaceModel
from .affine import AffineProcess, AffineObservations
import torch
from torch import distributions as dists


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


class LinearGaussianObservations(StateSpaceModel):
    def __init__(self, hidden, a=1., scale=1.):
        """
        Implements a State Space model that's linear in the observation equation but has arbitrary dynamics in the
        state process.
        :param hidden: The hidden dynamics
        :type hidden: AffineProcess
        :param a: The A-matrix, must be constant (currently)
        :type a: torch.Tensor|float|dists.Distribution
        :param scale: The variance of the observations, can be constant or learnable. Currently assumes that all
                      components are independent.
        :type scale: torch.Tensor|dists.Distribution|float
        """

        # ===== Convoluted way to decide number of dimensions ===== #
        is_1d = False
        if isinstance(a, dists.Distribution):
            dim = 1 if len(a.event_shape) < 2 else a.event_shape[0]
            is_1d = len(a.event_shape) == 1
        elif isinstance(a, float) or a.dim() < 2:
            dim = 1
            is_1d = (torch.tensor(a) if isinstance(a, float) else a).dim() == 1
        else:
            dim = a.shape[0]

        # ====== Define distributions ===== #
        n = dists.Normal(0., 1.) if dim < 2 else dists.Independent(dists.Normal(torch.zeros(dim), torch.ones(dim)), 1)
        scale = (scale,) if not isinstance(scale, (tuple, list)) else scale

        # ===== Determine propagator function ===== #
        if dim > 1:
            f = f_2d
        elif is_1d:
            f = f_1d
        else:
            f = f_0d

        observable = AffineObservations((f, g), (a, *scale), n)

        super().__init__(hidden, observable)