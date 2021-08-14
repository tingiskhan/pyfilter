from torch.distributions import Normal, Independent, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
import torch
from ...distributions import DistributionWrapper
from ...typing import ArrayType


def init_trans(module: "OrnsteinUhlenbeck", dist):
    kappa, gamma, sigma, initial = module.functional_parameters()

    initial_ = gamma if initial is None else initial

    return TransformedDistribution(dist, AffineTransform(initial_, sigma / (2 * kappa).sqrt()))


# TODO: Fix s.t. initial distribution is function of parameters
class OrnsteinUhlenbeck(AffineProcess):
    """
    Implements the Ornstein-Uhlenbeck process.
    """

    def __init__(self, kappa, gamma, sigma, ndim: int, dt: float, initial_state_mean: ArrayType = None, **kwargs):
        if ndim > 1:
            dist = DistributionWrapper(
                lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(ndim), scale=torch.ones(ndim)
            )
        else:
            dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        super().__init__(
            (self._f, self._g),
            (kappa, gamma, sigma, initial_state_mean),
            dist,
            dist,
            initial_transform=init_trans,
            **kwargs
        )
        self._dt = torch.tensor(dt)

    def _f(self, x, k, g, s, _):
        return g + (x.values - g) * torch.exp(-k * self._dt)

    def _g(self, x, k, g, s, _):
        return s / (2 * k).sqrt() * (1 - torch.exp(-2 * k * self._dt)).sqrt()
