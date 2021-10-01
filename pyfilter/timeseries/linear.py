from .model import StateSpaceModel
from .observable import AffineObservations
import torch
from torch.distributions import Distribution, Normal, Independent
from typing import Union
from ..distributions import DistributionWrapper
from ..typing import ArrayType


def _f_0d(x, a, scale):
    return a * x.values


def _f_1d(x, a, scale):
    return _f_2d(x, a.unsqueeze(-2), scale).squeeze(-1)


def _f_2d(x, a, scale):
    return torch.matmul(a, x.values.unsqueeze(-1)).squeeze(-1)


def _g(x, a, *scale):
    return scale[-1] if len(scale) == 1 else scale


def _get_shape(a):
    is_1d = False
    if isinstance(a, Distribution):
        dim = a.event_shape
        is_1d = len(a.event_shape) == 1
    elif isinstance(a, float) or a.dim() < 2:
        dim = torch.Size([])
        is_1d = (torch.tensor(a) if isinstance(a, float) else a).dim() <= 1
    else:
        dim = a.shape[:1]

    return dim, is_1d


class LinearObservations(StateSpaceModel):
    """
    Defines a class of observation dynamics where the observed variable is a linear combination of the states, i.e.

        Y = a * x + scale * base_dist,
    """

    def __init__(self, hidden, a: ArrayType, scale: ArrayType, base_dist):

        dim, is_1d = _get_shape(a)

        if base_dist().event_shape != dim:
            raise ValueError("The distribution is not of correct shape!")

        if not is_1d:
            f = _f_2d
        elif is_1d and hidden.n_dim > 0:
            f = _f_1d
        else:
            f = _f_0d

        observable = AffineObservations((f, _g), (a, scale), base_dist)

        super().__init__(hidden, observable)


class LinearGaussianObservations(LinearObservations):
    """
    Implements an SSM of type `LinearObservations` where the `base_dist` corresponds to a Gaussian distribution.
    """

    def __init__(self, hidden, a=1.0, scale=1.0, **kwargs):
        dim, is_1d = _get_shape(a)

        if is_1d:
            n = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            n = DistributionWrapper(
                lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(dim), scale=torch.ones(dim)
            )

        super().__init__(hidden, a, scale, n, **kwargs)
