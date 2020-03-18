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
        Implements a one-step Euler-Maruyama model, similar to PyMC3. I.e. where we perform one iteration of the
        following recursion
            dX[t] = a(X[t-1]) * dt + b(X[t-1]) * dW[t]
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


class GeneralEulerMaruyama(StochasticDifferentialEquation):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, prop_state, **kwargs):
        """
        The Euler-Maruyama discretization scheme for stochastic differential equations of general type. I.e. you have
        full freedom for specifying the model. The recursion is defined as
            dX[t] = prop_state(x[t-1], a(X[t-1], dt), b(x[t-1], dt))
        :param dynamics: A tuple of callable, where the last argument should be `dt`
        :type dynamics: tuple[callable]
        :param prop_state: The function for propagating the state. Should take as input (x, f(x), g(x))
        :type prop_state: callable
        """

        super().__init__(dynamics, theta, init_dist, increment_dist, dt)
        self._ns = kwargs.pop('num_steps', 1)
        self._prop_state = prop_state

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

        return self._prop_state(x, self.f(x, *self.theta_vals, self._dt), self.g(x, *self.theta_vals, self._dt))

    def _propagate(self, x, as_dist=False):
        for i in range(self._ns):
            x = self.prop_state(x)

        if not as_dist:
            return x

        return Empirical(x)


class AffineEulerMaruyama(GeneralEulerMaruyama):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, **kwargs):
        """
        Euler Maruyama method for SDEs of affine nature. A generalization of OneStepMaruyama that allows multiple
        recursions. The difference between this class and GeneralEulerMaruyama is that you need not specify prop_state
        as that is assumed to follow the structure of OneStepEulerMaruyama.
        :param dynamics: A tuple of callable. Should _not_ include `dt` as the last argument
        :type dynamics: tuple[callable]
        """

        def _f(x, *args):
            return x + dynamics[0](x, *args) * self._dt

        super().__init__((_f, dynamics[-1]), theta, init_dist, increment_dist, dt, self._f, **kwargs)

    @tensor_caster
    def prop_state(self, x):
        """
        Helper method for propagating the state.
        :param x: The state
        :type x: torch.Tensor
        :return: Tensor
        :rtype: torch.Tensor
        """

        return self._prop_state(x, self.f(x, *self.theta_vals), self.g(x, *self.theta_vals))

    def _f(self, x, f, g):
        return self._define_transdist(f, g).sample()

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            m, s = self.mean_scale(x)
            x = m + s * u

        return x
