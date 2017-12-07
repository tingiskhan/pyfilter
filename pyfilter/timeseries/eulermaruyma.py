from .meta import Base
from math import sqrt
from ..utils.utils import resizer


class EulerMaruyma(Base):
    def __init__(self, initial, funcs, theta, noise, dt=1):
        super().__init__(initial, funcs, theta, noise)

        self.dt = dt

    def mean(self, x, *args, params=None):
        return x + resizer(self.f(x, *args, *(params or self.theta))) * self.dt

    def scale(self, x, *args, params=None):
        return resizer(self.g(x, *args, *(params or self.theta))) * sqrt(self.dt)
