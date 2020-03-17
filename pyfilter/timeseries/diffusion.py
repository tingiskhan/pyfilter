from .affine import AffineProcess
import torch
from abc import ABC
from torch.distributions import Distribution, Normal, Independent
from ..utils import Empirical
from .utils import tensor_caster


class StochasticDifferentialEquation(AffineProcess, ABC):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, num_steps=1):
        """
        Base class for stochastic differential equations. Note that the incremental distribution should include `dt`,
        as in for normal distributions the variance should be `dt`, and same for other.
        :param dynamics: The dynamics, tuple of functions
        :type dynamics: tuple[callable]
        :param dt: The step size
        :type dt: float
        :param num_steps: The number of integration steps, such that we simulate `num_steps` with step size `dt`
        :type num_steps: int
        """

        self._dt = dt
        self._ns = num_steps

        super().__init__(dynamics, theta, init_dist, increment_dist)


class OneStepEulerMaruyma(StochasticDifferentialEquation):
    def __init__(self, funcs, theta, initial_dist, inc_dist, dt):
        """
        Implements a one-step Euler-Maruyama model, similar to PyMC3.
        :param dt: The step-size to use in the approximation.
        :type dt: float|torch.Tensor
        """

        def f(x, *args):
            return x + funcs[0](x, *args) * self._dt

        def g(x, *args):
            return funcs[1](x, *args)

        super().__init__((f, g), theta, initial_dist, inc_dist, dt=dt, num_steps=1)


class OrnsteinUhlenbeck(StochasticDifferentialEquation):
    def __init__(self, kappa, gamma, sigma, ndim, dt):
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

        def f(x, reversion, level, std):
            return level + (x - level) * torch.exp(-reversion * self._dt)

        def g(x, reversion, level, std):
            return std / (2 * reversion).sqrt() * (1 - torch.exp(-2 * reversion * self._dt)).sqrt()

        if ndim > 1:
            dist = Independent(Normal(torch.zeros(ndim), torch.ones(ndim)), 1)
        else:
            dist = Normal(0., 1)

        super().__init__((f, g), (kappa, gamma, sigma), dist, dist, dt=dt, num_steps=1)


class GeneralEulerMaruyama(OneStepEulerMaruyma):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, **kwargs):
        """
        The Euler-Maruyama discretization scheme for stochastic differential equations.
        """

        super().__init__(dynamics, theta, init_dist, increment_dist, dt)
        self._ns = kwargs.pop('num_steps', 1)

    def _propagate_u(self, x, u):
        raise NotImplementedError()

    @tensor_caster
    def prop_state(self, x):
        """
        Helper method for propagating the state.
        :param x: The state
        :type x: torch.Tensor
        :return: Tensor
        :rtype: torch.Tensor
        """

        return self._prop_state(x)

    def _prop_state(self, x):
        raise NotImplementedError()

    def _propagate(self, x, as_dist=False):
        for i in range(self._ns):
            x = self.prop_state(x)

        if not as_dist:
            return x

        return Empirical(x)


class EulerMaruyama(GeneralEulerMaruyama):
    """
    Euler Maruyama method for SDEs of affine nature.
    """

    def _prop_state(self, x):
        return self._define_transdist(*self.mean_scale(x)).sample()

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            m, s = self.mean_scale(x)
            x = m + s * u

        return x
