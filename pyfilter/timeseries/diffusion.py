from .affine import AffineProcess
import torch
from abc import ABC
from torch.distributions import Distribution, Normal, Independent


class StochasticDifferentialEquation(AffineProcess, ABC):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, num_steps=1):
        """
        Base class for stochastic differential equations.
        :param dynamics: The dynamics, tuple of functions
        :type dynamics: tuple[callable]
        :param dt: The step size
        :type dt: float
        :param num_steps: The number of integration steps, such that we simulate `num_steps` with step size `dt`
        :type num_steps: int
        """

        self._dt = torch.tensor(dt)
        self._sqdt = self._dt.sqrt()
        self._ns = num_steps

        super().__init__(dynamics, theta, init_dist, increment_dist)


class OneStepEulerMaruyma(StochasticDifferentialEquation):
    def __init__(self, funcs, theta, initial_dist, inc_dist, dt=1.):
        """
        Implements a one-step Euler-Maruyama model, similar to PyMC3.
        :param dt: The step-size to use in the approximation.
        :type dt: float|torch.Tensor
        """

        def f(x, *args):
            return x + funcs[0](x, *args) * self._dt

        def g(x, *args):
            return funcs[1](x, *args) * self._sqdt

        super().__init__((f, g), theta, initial_dist, inc_dist, dt=dt, num_steps=1)


class OrnsteinUhlenbeck(StochasticDifferentialEquation):
    def __init__(self, kappa, gamma, sigma, ndim, dt=1.):
        """
        Implements the Ornstein-Uhlenbeck process.
        :param kappa: The reversion parameter
        :type kappa: torch.Tensor|float|Distribution
        :param gamma: The mean parameter
        :type gamma: torch.Tensor|float|Distribution
        :param sigma: The standard deviation
        :type sigma: torch.Tensor|float|Distribution
        :param ndim: The number of dimensions for the Brownian motion
        :type ndim: int
        """

        if ndim > 1:
            dist = Independent(Normal(torch.zeros(ndim), torch.ones(ndim)), 1)
        else:
            dist = Normal(0., 1)

        super().__init__((self._f, self._g), (kappa, gamma, sigma), dist, dist, dt=dt, num_steps=1)

    def _f(self, x, reversion, level, std):
        return level + (x - level) * torch.exp(-reversion * self._dt)

    def _g(self, x, reversion, level, std):
        return std / (2 * reversion).sqrt() * (1 - torch.exp(-2 * reversion * self._dt)).sqrt()


# TODO: Works for all distributions where the variance of the incremental distribution can be constructed as a product
class EulerMaruyama(OneStepEulerMaruyma):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, **kwargs):
        """
        The Euler-Maruyama discretization scheme for stochastic differential equations.
        """

        super().__init__(dynamics, theta, init_dist, increment_dist, dt)
        self._ns = kwargs.pop('num_steps', 1)

    def _propagate(self, x, as_dist=False):
        if as_dist:
            raise ValueError(f'Does not work for {self.__class__.__name__}!')

        for i in range(self._ns):
            dist = self._define_transdist(*self.mean_scale(x))
            x = dist.sample()

        return x

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            m, s = self.mean_scale(x)
            x = m + s * u

        return x

