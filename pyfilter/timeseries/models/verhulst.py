from ..diffusion import AffineEulerMaruyama
from torch.distributions import Normal, TransformedDistribution, AbsTransform
from .ou import init_trans
from math import sqrt
from ...distributions import DistributionWrapper


def f(x, k, g, s):
    return k * (g - x.state) * x.state


def g_(x, k, g, s):
    return s * x.state


def init_transform(dist, k, g, s):
    dist = init_trans(dist, k, g, s)

    return TransformedDistribution(dist, AbsTransform())


class Verhulst(AffineEulerMaruyama):
    def __init__(self, kappa, gamma, sigma, dt, num_steps, **kwargs):
        """
        Defines a Verhulst process.

        :param kappa: The reversion parameter
        :param gamma: The mean parameter
        :param sigma: The standard deviation
        """

        super().__init__(
            (f, g_),
            (kappa, gamma, sigma),
            DistributionWrapper(Normal, loc=0.0, scale=1.0),
            DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt)),
            dt=dt,
            num_steps=num_steps,
            initial_transform=init_transform,
            **kwargs
        )
