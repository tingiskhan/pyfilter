from .affine import AffineProcess, _define_transdist
from .base import StochasticProcess
import torch
from abc import ABC
from torch.distributions import Distribution, Normal, Independent
from ..utils import Empirical


class StochasticDifferentialEquation(StochasticProcess, ABC):
    def __init__(self, theta, init_dist, increment_dist, dt, num_steps=1):
        """
        Base class for stochastic differential equations. Note that the incremental distribution should include `dt`,
        as in for normal distributions the variance should be `dt`, and same for other.
        :param dt: The step size
        :type dt: float
        :param num_steps: The number of integration steps, such that we simulate `num_steps` with step size `dt`
        :type num_steps: int
        """

        super().__init__(theta, init_dist, increment_dist)

        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt
        self._ns = num_steps


class OneStepEulerMaruyma(AffineProcess):
    def __init__(self, dynamics, theta, initial_dist, inc_dist, dt):
        """
        Implements a one-step Euler-Maruyama model, similar to PyMC3. I.e. where we perform one iteration of the
        following recursion
            dX[t] = a(X[t-1]) * dt + b(X[t-1]) * dW[t]
        :param dynamics: The dynamics, tuple of functions
        :type dynamics: tuple[callable]
        :param dt: The step-size to use in the approximation.
        :type dt: float|torch.Tensor
        """

        def f(x, *args):
            return x + dynamics[0](x, *args) * dt

        def g(x, *args):
            return dynamics[1](x, *args)

        super().__init__((f, g), theta, initial_dist, inc_dist)


class OrnsteinUhlenbeck(AffineProcess):
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
            return level + (x - level) * torch.exp(-reversion * dt)

        def g(x, reversion, level, std):
            return std / (2 * reversion).sqrt() * (1 - torch.exp(-2 * reversion * dt)).sqrt()

        if ndim > 1:
            dist = Independent(Normal(torch.zeros(ndim), torch.ones(ndim)), 1)
        else:
            dist = Normal(0., 1)

        super().__init__((f, g), (kappa, gamma, sigma), dist, dist)


class GeneralEulerMaruyama(StochasticDifferentialEquation):
    def __init__(self, theta, init_dist, dt, prop_state, num_steps, **kwargs):
        """
        The Euler-Maruyama discretization scheme for stochastic differential equations of general type. I.e. you have
        full freedom for specifying the model. The recursion is defined as
            X[t + 1] = prop_state(X[t], *parameters, dt)
        :param prop_state: The function for propagating the state. Should take as input (x, *parameters, dt)
        :type prop_state: callable
        """

        inc_dist = kwargs.pop('increment_dist', init_dist)
        super().__init__(theta, init_dist, inc_dist, dt, num_steps)
        self._prop_state = prop_state

    def _propagate(self, x, as_dist=False):
        for i in range(self._ns):
            x = self._prop_state(x, *self.theta_vals, dt=self._dt)

        if not as_dist:
            return x

        return Empirical(x)


class AffineEulerMaruyama(GeneralEulerMaruyama):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, **kwargs):
        """
        Euler Maruyama method for SDEs of affine nature. A generalization of OneStepMaruyama that allows multiple
        recursions. The difference between this class and GeneralEulerMaruyama is that you need not specify prop_state
        as it is assumed to follow the structure of OneStepEulerMaruyama.
        :param dynamics: A tuple of callable. Should _not_ include `dt` as the last argument
        :type dynamics: tuple[callable]
        """

        super().__init__(theta, init_dist, increment_dist=increment_dist, dt=dt, prop_state=self._prop, **kwargs)
        self.f, self.g = dynamics

    def _prop(self, x, *params, dt):
        f = self.f(x, *params) * dt
        g = self.g(x, *params)

        return _define_transdist(x + f, g, self.increment_dist, self.ndim).sample()

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            f = self.f(x, *self.theta_vals) * self._dt
            g = self.g(x, *self.theta_vals)

            x += f + g * u

        return x
