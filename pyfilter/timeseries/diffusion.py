import torch
from abc import ABC
from typing import Tuple
from .affine import AffineProcess, MeanOrScaleFun, _define_transdist
from .base import ParameterizedBase
from ..distributions import DistributionWrapper
from .typing import DiffusionFunction


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


class StochasticDifferentialEquation(ParameterizedBase, ABC):
    """
    Base class for stochastic differential equations.
    """

    def __init__(self, parameters, initial_dist: DistributionWrapper, dt: float, num_steps=1, **kwargs):
        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)

        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt
        self._ns = num_steps


class EulerMaruyama(StochasticDifferentialEquation):
    """
    The Euler-Maruyama discretization scheme for stochastic differential equations. The recursion is defined as:
    X[t + 1] = prop_state(X[t], dt, *parameters)
    """

    def __init__(
            self, prop_state: DiffusionFunction,
            parameters,
            initial_dist: DistributionWrapper,
            dt,
            num_steps,
            **kwargs
    ):
        super().__init__(parameters, initial_dist, dt, num_steps, **kwargs)
        self._propagator = prop_state

    def forward(self, x, time_increment=1.0):
        for i in range(self._ns):
            x = super(EulerMaruyama, self).forward(x, time_increment=self._dt)

        return x

    propagate = forward

    def build_density(self, x):
        return self._propagator(x, self._dt, *self.functional_parameters())


class AffineEulerMaruyama(EulerMaruyama):
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
        super().__init__(self._prop, parameters, initial_dist, dt=dt, **kwargs)
        self.f, self.g = dynamics
        self.increment_dist = increment_dist

    def _prop(self, x, dt, *params):
        f = self.f(x, *params) * dt
        g = self.g(x, *params)

        return _define_transdist(x.values + f, g, self.n_dim, self.increment_dist())

    def propagate_conditional(self, x, u, parameters=None):
        params = parameters or self.functional_parameters()
        for i in range(self._ns):
            f = self.f(x, *params) * self._dt
            g = self.g(x, *params)

            x = self.propagate_state(x.values + f + g * u, x, self._dt)

        return x.state

    def prop_apf(self, x):
        x_t = x
        for i in range(self._ns):
            f = self.f(x_t, *self.functional_parameters()) * self._dt
            x_t = self.propagate_state(x_t.state + f, x_t, self._dt)

        return self.propagate_state(x_t.state, x, self._ns * self._dt)
