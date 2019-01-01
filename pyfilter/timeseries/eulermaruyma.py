from .base import BaseModel
import torch


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
        self.dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt
        self._sqdt = dt.sqrt()

    def mean(self, x):
        return x + self.f_val(x) * self.dt

    def scale(self, x, params=None):
        return self.g_val(x) * self._sqdt