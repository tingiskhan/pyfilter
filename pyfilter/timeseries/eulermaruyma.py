from .affine import AffineModel
import torch
from torch.distributions import Normal, Independent, Distribution


# TODO: Add integration step as well
class EulerMaruyma(AffineModel):
    def __init__(self, initial, funcs, theta, dt=1., ndim=1):
        """
        Implements the Euler-Maruyama scheme.
        :param ndim: The number of dimensions
        :type ndim: int
        :param dt: The step-size to use in the approximation. If `dt=1`, is basically AR process
        :type dt: float|torch.Tensor
        """

        self.dt = torch.tensor(float(dt)) if not isinstance(dt, torch.Tensor) else dt
        self._sqdt = self.dt.sqrt()

        if ndim > 1:
            dist = Independent(Normal(torch.zeros(ndim), torch.ones(ndim)), 1)
        else:
            dist = Normal(0., 1)

        super().__init__(initial, funcs, theta, (dist, dist))

    def mean(self, x):
        return x + self.f_val(x) * self.dt

    def scale(self, x):
        return self.g_val(x) * self._sqdt


def _fh0(reversion, level, std):
    return level


def _gh0(reversion, level, std):
    return std / torch.sqrt(2 * reversion)


class OrnsteinUhlenbeck(EulerMaruyma):
    def __init__(self, kappa, gamma, sigma, dt=1.):
        """
        Implements the Ornstein-Uhlenbeck process.
        :param kappa: The reversion parameter
        :type kappa: torch.Tensor|float|Distribution
        :param gamma: The mean parameter
        :type gamma: torch.Tensor|float|Distribution
        :param sigma: The standard deviation
        :type sigma: torch.Tensor|float|Distribution
        """
        super().__init__((_fh0, _gh0), (self._f, self._g), (kappa, gamma, sigma), dt=dt, ndim=1)

    def mean(self, x):
        return self.f_val(x)

    def scale(self, x):
        return self.g_val(x)

    def _f(self, x, reversion, level, std):
        return level + (x - level) * torch.exp(-reversion * self.dt)

    def _g(self, x, reversion, level, std):
        return std / (2 * reversion).sqrt() * (1 - torch.exp(-2 * reversion * self.dt)).sqrt()