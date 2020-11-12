from ..diffusion import AffineEulerMaruyama
from torch.distributions import Normal, LogNormal


class Verhulst(AffineEulerMaruyama):
    def __init__(self, kappa, gamma, sigma, **kwargs):
        """
        Defines a Verhulst process.
        :param kappa: The reversion parameter
        :param gamma: The mean parameter
        :param sigma: The standard deviation
        """

        def f(x, k, g, s):
            return k * (g - x) * x

        def g_(x, k, g, s):
            return s * x

        super().__init__((f, g_), (kappa, gamma, sigma), LogNormal(0.0, 1.0), Normal(0.0, 1.0), **kwargs)
