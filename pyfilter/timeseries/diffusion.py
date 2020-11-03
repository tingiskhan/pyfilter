from .affine import AffineProcess, _define_transdist
from .process import StochasticProcess
import torch
from abc import ABC
from torch.distributions import Distribution
from ..utils import Empirical
from typing import Callable, Tuple


class StochasticDifferentialEquation(StochasticProcess, ABC):
    def __init__(self, parameters, initial_dist, increment_dist, dt: float, num_steps=1):
        """
        Base class for stochastic differential equations. Note that the incremental distribution should include `dt`,
        as in for normal distributions the variance should be `dt`, and same for other.
        :param dt: The step size
        :param num_steps: The number of integration steps, such that we simulate `num_steps` with step size `dt`
        """

        super().__init__(parameters, initial_dist, increment_dist)

        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt
        self._ns = num_steps


class OneStepEulerMaruyma(AffineProcess):
    def __init__(self, dynamics, parameters, initial_dist, increment_dist, dt: float, initial_transform=None):
        """
        Implements a one-step Euler-Maruyama model, similar to PyMC3. I.e. where we perform one iteration of the
        following recursion
            dX[t] = a(X[t-1]) * dt + b(X[t-1]) * dW[t]
        :param dynamics: The dynamics, tuple of functions
        :param dt: The step-size to use in the approximation.
        """

        def f(x: torch.Tensor, *args: object) -> torch.Tensor:
            return x + dynamics[0](x, *args) * dt

        def g(x: torch.Tensor, *args: object) -> torch.Tensor:
            return dynamics[1](x, *args)

        super().__init__((f, g), parameters, initial_dist, increment_dist, initial_transform=initial_transform)


class GeneralEulerMaruyama(StochasticDifferentialEquation):
    def __init__(self, parameters, initial_dist: Distribution, dt,
                 prop_state: Callable[[torch.Tensor, Tuple[object, ...], float], torch.Tensor], num_steps, **kwargs):
        """
        The Euler-Maruyama discretization scheme for stochastic differential equations of general type. I.e. you have
        full freedom for specifying the model. The recursion is defined as
            X[t + 1] = prop_state(X[t], *parameters, dt)
        :param prop_state: The function for propagating the state. Should take as input (x, *parameters, dt)
        """

        inc_dist = kwargs.pop('increment_dist', initial_dist)
        super().__init__(parameters, initial_dist, inc_dist, dt, num_steps)
        self._prop_state = prop_state

    def define_density(self, x, u=None):
        for i in range(self._ns):
            x = self._prop_state(x, *self.parameter_views, dt=self._dt)

        return Empirical(x)


class AffineEulerMaruyama(GeneralEulerMaruyama):
    def __init__(self, dynamics: Tuple[Callable[[torch.Tensor, Tuple[object, ...]], torch.Tensor], ...], parameters,
                 initial_dist, increment_dist, dt, **kwargs):
        """
        Euler Maruyama method for SDEs of affine nature. A generalization of OneStepMaruyama that allows multiple
        recursions. The difference between this class and GeneralEulerMaruyama is that you need not specify prop_state
        as it is assumed to follow the structure of OneStepEulerMaruyama.
        :param dynamics: A tuple of callable. Should _not_ include `dt` as the last argument
        """

        super().__init__(parameters, initial_dist, increment_dist=increment_dist, dt=dt, prop_state=self._prop, **kwargs)
        self.f, self.g = dynamics

    def _prop(self, x, *params, dt):
        f = self.f(x, *params) * dt
        g = self.g(x, *params)

        return _define_transdist(x + f, g, self.increment_dist, self.ndim).sample()

    def _propagate_u(self, x, u):
        for i in range(self._ns):
            f = self.f(x, *self.parameter_views) * self._dt
            g = self.g(x, *self.parameter_views)

            x += f + g * u

        return x
