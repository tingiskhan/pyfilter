from .base import AffineModel
import torch
from torch.distributions import Normal, Independent
from .parameter import Parameter


class EulerMaruyma(AffineModel):
    def __init__(self, initial, funcs, theta, dt=1., ndim=1):
        """
        Implements the Euler-Maruyama scheme.
        :param ndim: The number of dimensions
        :type ndim: int
        :param dt: The step-size to use in the approximation. If `dt=1`, is basically AR process
        :type dt: float|torch.Tensor
        """

        if ndim > 1:
            dist = Independent(Normal(torch.zeros(ndim), torch.ones(ndim)), 1)
        else:
            dist = Normal(0., 1)

        super().__init__(initial, funcs, theta, (dist, dist))
        self.dt = torch.tensor(float(dt)) if not isinstance(dt, torch.Tensor) else dt
        self._sqdt = self.dt.sqrt()

    def mean(self, x):
        return x + self.f_val(x) * self.dt

    def scale(self, x, params=None):
        return self.g_val(x) * self._sqdt


def _fh0(reversion, level, std):
    return level


def _gh0(reversion, level, std):
    return std / torch.sqrt(2 * reversion)


class OrnsteinUhlenbeck(EulerMaruyma):
    def __init__(self, kappa, gamma, sigma, dt=1., ndim=1):
        """
        Implements the Ornstein-Uhlenbeck process.
        :param kappa: The reversion parameter
        :type kappa: torch.Tensor|float|Parameter
        :param gamma: The mean parameter
        :type gamma: torch.Tensor|float|Parameter
        :param sigma: The standard deviation
        :type sigma: torch.Tensor|float|Parameter
        """
        super().__init__((_fh0, _gh0), (self._f, self._g), (kappa, gamma, sigma), dt=dt, ndim=ndim)

    def _f(self, x, reversion, level, std):
        return level + (x - level) * torch.exp(-reversion * self.dt) - x

    def _g(self, x, reversion, level, std):
        return std / (2 * reversion).sqrt() * (1 - torch.exp(-2 * reversion * self.dt)).sqrt()