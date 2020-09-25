from .affine import AffineProcess
from typing import Callable
import torch


class AffineObservations(AffineProcess):
    def __init__(self, funcs, theta, increment_dist):
        """
        Class for defining model with affine dynamics in the observable process.
        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(funcs, theta, None, increment_dist)
        self._covariate = None

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")

    def add_covariate(self, f: Callable[[torch.Tensor], torch.Tensor]):
        """
        Adds a covariate function to the observable density, such that
        Y_t = f(U_t) + \mu(X_t) + \sigma(X_t) * W_t
        :param f: The covariate function
        """

        self._covariate = f

        return self

    def define_density(self, x, u=None):
        loc, scale = self._mean_scale(x)

        if (u is not None) and (self._covariate is not None):
            loc += self._covariate(u, *self.theta_vals)

        return self._define_transdist(loc, scale)
