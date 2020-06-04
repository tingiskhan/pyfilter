from ..diffusion import AffineEulerMaruyama
from torch.distributions import Normal


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

        dist = Normal(0., 1.)

        super().__init__((f, g_), (kappa, gamma, sigma), dist, dist, **kwargs)
