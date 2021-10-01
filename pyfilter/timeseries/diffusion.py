import torch
from abc import ABC
from typing import Tuple
from torch.distributions import Normal, Independent
import math
from .affine import AffineProcess, MeanOrScaleFun
from .stochasticprocess import StructuralStochasticProcess
from .typing import DiffusionFunction
from ..distributions import DistributionWrapper
from ..typing import ArrayType
from ..constants import EPS


class OneStepEulerMaruyma(AffineProcess):
    """
    Implements a one-step Euler-Maruyama model, similar to PyMC3. I.e. where we perform one iteration of the
    following recursion:
        .. math::
            X_{t+1} = X_t + a(X_t) \Delta t + b(X_t) \cdot \Delta W_t
    """

    def __init__(self, funcs, parameters, initial_dist, increment_dist, dt: float, **kwargs):
        """
        Initializes the ``OneStepEulerMaruyma`` class.

        Args:
            funcs: See base.
            parameters: See base.
            initial_dist: See base.
            increment_dist: See base. However, do not that you need to include the :math:`\Delta t` term yourself in
               the ``DistributionWrapper`` class.
            dt: The time delta to use.
        """

        super().__init__(funcs, parameters, initial_dist, increment_dist, **kwargs)
        self.dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def mean_scale(self, x, parameters=None):
        drift, diffusion = super(OneStepEulerMaruyma, self).mean_scale(x, parameters=parameters)

        return x.values + drift * self.dt, diffusion


class StochasticDifferentialEquation(StructuralStochasticProcess, ABC):
    """
    Abstract base class for stochastic differential equations, i.e. stochastic processes given by
        .. math::
            dX_{t+1} = h(X_t, \\theta, dt),

    where :math:`\\theta` is the parameter set, and :math:`h` the dynamics of the SDE.
    """

    def __init__(self, parameters, initial_dist: DistributionWrapper, dt: float, **kwargs):
        """
        Initializes the ``StochasticDifferentialEquation``.

        Args:
            parameters: See base.
            initial_dist: See base.
            dt: The time discretization to use.
            kwargs: See base.
        """

        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)

        self.dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt


class DiscretizedStochasticDifferentialEquation(StochasticDifferentialEquation):
    """
    Defines a discretized stochastic differential equation, in which the next state :math:`X_{t+\Delta t}` is given by
        .. math::
            X_{t+\Delta t} = h(X_t, \\theta, \Delta t).

    This e.g. encompasses the Euler-Maruyama and Milstein schemes.
    """

    def __init__(self, prop_state: DiffusionFunction, parameters, initial_dist: DistributionWrapper, dt, **kwargs):
        """
        Initializes the ``DiscretizedStochasticDifferentialEquation`` class.

        Args:
            prop_state: Corresponds to the function :math:`h`.
            parameters: See base.
            initial_dist: See base.
            dt: See base.
            kwargs: See base.
        """

        super().__init__(parameters, initial_dist, dt, **kwargs)
        self._propagator = prop_state

    def build_density(self, x):
        return self._propagator(x, self.dt, *self.functional_parameters())


class AffineEulerMaruyama(AffineProcess, StochasticDifferentialEquation):
    """
    Defines the Euler-Maruyama scheme for an SDE of affine nature, i.e. in which the dynamics are given by the
    functional pair of the drift and diffusion, such that we have
        .. math::
            X_{t+\Delta t} = X_t + f_\\theta(X_t) \Delta t + g_\\theta(X_t) \Delta W_t,

    where :math:`W_t` is an arbitrary random variable from which we can sample.
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
        """
        Initializes the ``AffineEulerMaruyama`` class.

        Args:
            dynamics: Tuple of callables that define the drift and diffusion respectively.
            parameters: See base.
            initial_dist: See base.
            increment_dist: See ``AffineProcess``.
            dt: See base.
            kwargs: See base.
        """

        super(AffineEulerMaruyama, self).__init__(
            dynamics, parameters, initial_dist, dt=dt, increment_dist=increment_dist, **kwargs
        )

    def mean_scale(self, x, parameters=None):
        params = parameters or self.functional_parameters()
        return x.values + self.f(x, *params) * self.dt, self.g(x, *params)


class Euler(AffineEulerMaruyama):
    """
    Implements the standard Euler scheme for an ODE by reframing the model into a stochastic process using low variance
    Normal distributed noise in the state process. That is, given an ODE of the form
        .. math::
            \\frac{dx}{dt} = f_\\theta(x_t),

    we recast the model into
        .. math::
            X_{t + \Delta t} = X_t + f_\\theta(X_t) \Delta t + W_t,
            _
    where :math:`W_t` is a zero mean Gaussian distribution with a tunable standard deviation :math:`\sigma_{tune}`. In
    the limit that :math:`\sigma_{tune} \\rightarrow 0` we get the standard Euler scheme for ODEs. Thus, the higher the
    value of :math:`\sigma_{tune}`, the more we allow to deviate from the original model.

    See: https://arxiv.org/abs/2011.09718?context=stat
    """

    def __init__(
        self, dynamics: MeanOrScaleFun, parameters, initial_values: ArrayType, dt, tuning_std: float = 1.0, **kwargs
    ):
        """
        Initializes the ``Euler`` class.

        Args:
            dynamics: Corresponds to the function :math:`f` in the main docs.
            parameters: See base.
            initial_values: The initial value(s) of the ODE. Can be a prior as well.
            dt: See base.
            tuning_std: The tuning standard deviation of the Gaussian distribution.
            kwargs: See base.
        """

        scale = torch.ones(
            initial_values.shape if isinstance(initial_values, torch.Tensor) else initial_values().event_shape
        )

        iv = DistributionWrapper(lambda **u: Independent(Normal(**u), 1), loc=initial_values, scale=EPS * scale,)

        event_shape = iv().event_shape
        if len(event_shape) == 0:
            dist = DistributionWrapper(Normal, loc=0.0, scale=math.sqrt(dt) * tuning_std)
        else:
            dist = DistributionWrapper(
                lambda **u: Independent(Normal(**u), 1),
                loc=torch.zeros(event_shape),
                scale=tuning_std * math.sqrt(dt) * torch.ones(event_shape),
            )

        super().__init__((dynamics, lambda *args: 1.0), parameters, iv, dist, dt, **kwargs)


class RungeKutta(Euler):
    """
    Same as ``Euler``, but instead of utilizing the Euler scheme, we use the `RK4 method`_.

    .. _`RK4 method`: https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """

    def mean_scale(self, x, parameters=None):
        params = parameters or self.functional_parameters()

        k1 = self.f(x, *params)
        k2 = self.f(x.propagate_from(time_increment=self.dt / 2, values=x.values + self.dt * k1 / 2), *params)
        k3 = self.f(x.propagate_from(time_increment=self.dt / 2, values=x.values + self.dt * k2 / 2), *params)
        k4 = self.f(x.propagate_from(time_increment=self.dt, values=x.values + self.dt * k3), *params)

        return x.values + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), self.g(x, *params)
