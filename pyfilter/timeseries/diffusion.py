from .base import StochasticProcess
from .affine import AffineProcess
from torch.distributions import Normal, Independent, Distribution
import torch


class StochasticDifferentialEquation(StochasticProcess):
    def __init__(self, dynamics, theta, init_dist, increment_dist, integrator):
        """
        Base class for stochastic differential equations.
        :param dynamics: The dynamics, tuple of functions
        :type dynamics: tuple[callable]
        :param integrator: The class to use for integration
        """
        super().__init__(theta, init_dist, increment_dist)

        self._integrator = integrator
        self.f, self.g = dynamics



