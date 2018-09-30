from .meta import BaseModel
from math import sqrt


class EulerMaruyma(BaseModel):
    def __init__(self, initial, funcs, theta, noise, dt=1):
        """
        Implements the Euler-Maruyama scheme.
        :param initial: See BaseModel
        :param funcs: See BaseModel
        :param theta: See BaseModel
        :param noise: See BaseModel
        :param dt: The step-size to use in the approximation. If `dt=1`, is basically AR process
        :type dt: float
        """

        super().__init__(initial, funcs, theta, noise)
        self.dt = sqrt(dt)

    def mean(self, x, params=None):
        return x + self.f(x, *(params or self.theta_vals)) * self.dt

    def scale(self, x, params=None):
        return self.g(x, *(params or self.theta_vals)) * sqrt(self.dt)