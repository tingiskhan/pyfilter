from ..diffusion import AffineEulerMaruyama
from torch.distributions import Normal, TransformedDistribution, AbsTransform
from .ou import init_trans
from math import sqrt
from ...distributions import DistributionWrapper
from ...typing import ArrayType


def f(x, k, g, s):
    return k * (g - x.values) * x.values


def g_(x, k, g, s):
    return s * x.values


def init_transform(module, dist):
    dist = init_trans(module, dist)

    return TransformedDistribution(dist, AbsTransform())


class Verhulst(AffineEulerMaruyama):
    """
    Implements a discretized Verhulst SDE with the following dynamics
        .. math::
            dX_t = \\kappa (\\gamma - X_t)X_t dt + \\sigma X_t dW_t, \n
            X_0 \\sim \\left | \\mathcal{N}(x_0, \\frac{\\sigma}{\\sqrt{2\\kappa}} \\right |,

    where :math:`\\kappa, \\gamma, \\sigma > 0`.
    """

    def __init__(self, reversion, mean, vol, dt, **kwargs):
        """
        Initializes the ``Verhulst`` class.

        Args:
            reversion: Corresponds to :math:`\\kappa`.
            mean: Corresponds to :math:`\\gamma`.
            vol: Corresponds to :math:`\\sigma`.
            kwargs: See base.
        """

        super().__init__(
            (f, g_),
            (reversion, mean, vol),
            DistributionWrapper(Normal, loc=0.0, scale=1.0),
            DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt)),
            dt=dt,
            initial_transform=init_transform,
            **kwargs
        )
