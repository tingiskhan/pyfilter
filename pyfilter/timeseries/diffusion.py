from .affine import AffineProcess
import torch
from abc import ABC


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


# TODO: Works for all distributions where the variance of the incremental distribution can be constructed as a product
class EulerMaruyama(StochasticDifferentialEquation):
    def __init__(self, dynamics, theta, init_dist, increment_dist, dt, **kwargs):
        """
        The Euler-Maruyama discretization scheme for .
        """

        def f(x, *args):
            return x + dynamics[0](x, *args) * self._dt

        def g(x, *args):
            return dynamics[1](x, *args) * self._sqdt

        super().__init__((f, g), theta, init_dist, increment_dist, dt, **kwargs)

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

