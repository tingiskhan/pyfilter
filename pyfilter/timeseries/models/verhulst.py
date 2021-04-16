from ..diffusion import AffineEulerMaruyama
from torch.distributions import Normal, TransformedDistribution, AbsTransform
from .ou import init_trans
from math import sqrt
from ...distributions import DistributionWrapper


def f(x, k, g, s):
    return k * (g - x.values) * x.values


def g_(x, k, g, s):
    return s * x.values


def init_transform(module, dist):
    dist = init_trans(module, dist)

    return TransformedDistribution(dist, AbsTransform())


class Verhulst(AffineEulerMaruyama):
    """
    Defines a Verhulst process.
    """

    def __init__(self, reversion, mean, vol, dt, num_steps, **kwargs):
        super().__init__(
            (f, g_),
            (reversion, mean, vol),
            DistributionWrapper(Normal, loc=0.0, scale=1.0),
            DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt)),
            dt=dt,
            num_steps=num_steps,
            initial_transform=init_transform,
            **kwargs
        )
