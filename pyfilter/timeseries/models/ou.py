from torch.distributions import Normal, Independent, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
import torch
from ...distributions import DistributionWrapper


def init_trans(dist, kappa, gamma, sigma):
    return TransformedDistribution(dist, AffineTransform(gamma, sigma / (2 * kappa).sqrt()))


# TODO: Fix s.t. initial distribution is function of parameters
class OrnsteinUhlenbeck(AffineProcess):
    def __init__(self, kappa, gamma, sigma, ndim: int, dt: float):
        """
        Implements the Ornstein-Uhlenbeck process.
        :param kappa: The reversion parameter
        :param gamma: The mean parameter
        :param sigma: The standard deviation
        :param ndim: The number of dimensions for the Brownian motion
        """

        if ndim > 1:
            dist = DistributionWrapper(
                lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(ndim), scale=torch.ones(ndim)
            )
        else:
            dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        super().__init__((self._f, self._g), (kappa, gamma, sigma), dist, dist, initial_transform=init_trans)
        self._dt = torch.tensor(dt)

    def _f(self, x, k, g, s):
        return g + (x.current - g) * torch.exp(-k * self._dt)

    def _g(self, x, k, g, s):
        return s / (2 * k).sqrt() * (1 - torch.exp(-2 * k * self._dt)).sqrt()
