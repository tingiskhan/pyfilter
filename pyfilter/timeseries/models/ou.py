from torch.distributions import Normal, Independent
from ..affine import AffineProcess
import torch


class OrnsteinUhlenbeck(AffineProcess):
    def __init__(self, kappa, gamma, sigma, ndim: int, dt: float):
        """
        Implements the Ornstein-Uhlenbeck process.
        :param kappa: The reversion parameter
        :param gamma: The mean parameter
        :param sigma: The standard deviation
        :param ndim: The number of dimensions for the Brownian motion
        """

        def f(x: torch.Tensor, reversion: object, level: object, std: object):
            return level + (x - level) * torch.exp(-reversion * dt)

        def g(x: torch.Tensor, reversion: object, level: object, std: object):
            return std / (2 * reversion).sqrt() * (1 - torch.exp(-2 * reversion * dt)).sqrt()

        if ndim > 1:
            dist = Independent(Normal(torch.zeros(ndim), torch.ones(ndim)), 1)
        else:
            dist = Normal(0., 1)

        super().__init__((f, g), (kappa, gamma, sigma), dist, dist)