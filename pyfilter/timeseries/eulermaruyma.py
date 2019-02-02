from .base import AffineModel
import torch
from torch.distributions import Normal, Independent


class EulerMaruyma(AffineModel):
    def __init__(self, initial, funcs, theta, dt=1., ndim=1):
        """
        Implements the Euler-Maruyama scheme.
        :param ndim: The number of dimensions
        :type ndim: int
        :param dt: The step-size to use in the approximation. If `dt=1`, is basically AR process
        :type dt: float
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