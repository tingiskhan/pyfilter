from .meta import Base
from math import sqrt
from ..utils.utils import resizer


class EulerMaruyma(Base):
    def __init__(self, initial, funcs, theta, noise, dt=1):
        """
        Implements the Euler-Maruyama scheme.
        :param initial: See Base
        :param funcs: See Base
        :param theta: See Base
        :param noise: See Base
        :param dt: The step-size to use in the approximation. If `dt=1`, is basically AR process
        :type dt: float
        """
        super().__init__(initial, funcs, theta, noise)

        self.dt = dt

    def mean(self, x, params=None):
        return x + resizer(self.f(x, *(params or self.theta))) * self.dt

    def scale(self, x, params=None):
        return resizer(self.g(x, *(params or self.theta))) * sqrt(self.dt)
