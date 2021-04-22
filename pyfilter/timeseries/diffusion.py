import torch
from abc import ABC
from typing import Tuple
from torch.distributions import Normal, Independent
import math
from .affine import AffineProcess, MeanOrScaleFun
from .stochasticprocess import ParameterizedStochasticProcess
from .typing import DiffusionFunction
from ..distributions import DistributionWrapper
from ..typing import ArrayType
from ..constants import EPS


class OneStepEulerMaruyma(AffineProcess):
    """
    Implements a one-step Euler-Maruyama model, similar to PyMC3. I.e. where we perform one iteration of the
    following recursion: dX[t] = a(X[t-1]) * dt + b(X[t-1]) * dW[t]
    """

    def __init__(self, funcs, parameters, initial_dist, increment_dist, dt: float, **kwargs):
        super().__init__(funcs, parameters, initial_dist, increment_dist, **kwargs)
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def mean_scale(self, x, parameters=None):
        drift, diffusion = super(OneStepEulerMaruyma, self).mean_scale(x, parameters=parameters)

        return x.values + drift * self._dt, diffusion


class StochasticDifferentialEquation(ParameterizedStochasticProcess, ABC):
    """
    Base class for stochastic differential equations.
    """

    def __init__(self, parameters, initial_dist: DistributionWrapper, dt: float, num_steps=1, **kwargs):
        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)

        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt
        self._num_steps = num_steps


class EulerMaruyama(StochasticDifferentialEquation):
    """
    The Euler-Maruyama discretization scheme for stochastic differential equations. The recursion is defined as:
    X[t + 1] = prop_state(X[t], dt, *parameters)
    """

    def __init__(
        self, prop_state: DiffusionFunction, parameters, initial_dist: DistributionWrapper, dt, num_steps, **kwargs
    ):
        super().__init__(parameters, initial_dist, dt, num_steps, **kwargs)
        self._propagator = prop_state

    def forward(self, x, time_increment=1.0):
        for i in range(self._num_steps):
            x = super(EulerMaruyama, self).forward(x, time_increment=self._dt)

        return x

    propagate = forward

    def build_density(self, x):
        return self._propagator(x, self._dt, *self.functional_parameters())


# TODO: Make subclass of AffineProcess as well?
class AffineEulerMaruyama(AffineProcess, StochasticDifferentialEquation):
    """
    Euler-Maruyama method for SDEs of affine nature. A generalization of `OneStepMaruyama` that allows multiple
    recursions. Note that the incremental distribution should include the `dt` term as this is not done automatically.
    """

    def __init__(
        self,
        dynamics: Tuple[MeanOrScaleFun, ...],
        parameters,
        initial_dist,
        increment_dist: DistributionWrapper,
        dt,
        **kwargs
    ):
        super(AffineEulerMaruyama, self).__init__(
            dynamics, parameters, initial_dist, dt=dt, increment_dist=increment_dist, **kwargs
        )

    def mean_scale(self, x, parameters=None):
        params = parameters or self.functional_parameters()
        return x.values + self.f(x, *params) * self._dt, self.g(x, *params)

    def forward(self, x, time_increment=1.0):
        for i in range(self._num_steps):
            x = super(AffineEulerMaruyama, self).forward(x, time_increment=self._dt)

        return x

    propagate = forward

    def propagate_conditional(self, x, u, parameters=None, time_increment=1.0):
        for i in range(self._num_steps):
            x = super(AffineEulerMaruyama, self).propagate_conditional(x, u, parameters, time_increment)

        return x


class Euler(AffineEulerMaruyama):
    """
    Implements the standard Euler scheme for an ODE by reframing the model into a stochastic process using low variance
    Normal distributed noise in the state process.

    See: https://arxiv.org/abs/2011.09718?context=stat
    """

    def __init__(
        self, dynamics: MeanOrScaleFun, parameters, initial_values: ArrayType, dt, tuning_std: float = 1.0, **kwargs
    ):
        scale = torch.ones(
            initial_values.shape if isinstance(initial_values, torch.Tensor) else initial_values().event_shape
        )

        iv = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1),
            loc=initial_values,
            scale=EPS * scale,
        )

        event_shape = iv().event_shape
        if len(event_shape) == 0:
            dist = DistributionWrapper(Normal, loc=0.0, scale=math.sqrt(dt))
        else:
            dist = DistributionWrapper(
                lambda **u: Independent(Normal(**u), 1),
                loc=torch.zeros(event_shape),
                scale=tuning_std * math.sqrt(dt) * torch.ones(event_shape),
            )

        super().__init__((dynamics, lambda *args: 1.0), parameters, iv, dist, dt, **kwargs)


class RungeKutta(Euler):
    """
    Implements the RK4 method in a similar way as `Euler`.
    """

    def mean_scale(self, x, parameters=None):
        params = parameters or self.functional_parameters()

        k1 = self.f(x, *params)
        k2 = self.f(x.propagate_from(time_increment=self._dt / 2, values=x.values + self._dt * k1 / 2), *params)
        k3 = self.f(x.propagate_from(time_increment=self._dt / 2, values=x.values + self._dt * k2 / 2), *params)
        k4 = self.f(x.propagate_from(time_increment=self._dt, values=x.values + self._dt * k3), *params)

        return x.values + self._dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), self.g(x, *params)
