from ..diffusion import AffineEulerMaruyama
from torch.distributions import Normal, Independent
import torch
from ...utils import concater
from math import sqrt


def f(x, beta, gamma, sigma):
    s = beta * x[..., 0] * x[..., 1]

    r = x[..., 1] * gamma
    i = s - r

    return concater(-s, i, r)


def g(x, beta, gamma, sigma):
    s = -sigma * x[..., 0] * x[..., 1]
    r = torch.zeros_like(s)

    return concater(s, -s, r)


class OneFactorSIR(AffineEulerMaruyama):
    def __init__(self, parameters, initial_dist, dt, **kwargs):
        """
        Implements a SIR model where the number of sick has been replaced with the fraction of sick people relative to
        the entire population.
        Model taken from this article: https://arxiv.org/pdf/2004.06680.pdf
        """

        if initial_dist.event_shape != torch.Size([3]):
            raise ValueError(f"Initial distribution must be of shape 3, but got: {initial_dist.event_shape}")

        increment_dist = Independent(Normal(torch.zeros(1), sqrt(dt) * torch.ones(1)), 1)
        super().__init__((f, g), parameters, initial_dist, increment_dist, dt, **kwargs)
