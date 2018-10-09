from .base import Proposal
from torch.distributions import Normal
from ..timeseries import LinearGaussianObservations as LGO
import torch


# TODO: Seems to work for 1D models currently, will need to extend to multidimensional
class LinearGaussianObservations(Proposal):
    def set_model(self, model):
        if not isinstance(model, LGO):
            raise ValueError('Model must be of instance {}'.format(LGO.__name__))

        self._model = model

        return self

    def draw(self, y, x, size=None, *args, **kwargs):
        h_var = self._model.hidden.scale(x) ** 2
        c = self._model.observable.theta_vals[0]
        o_var = self._model.observable.theta_vals[-1] ** 2

        cov = 1 / (1 / h_var + c ** 2 / o_var)
        m = cov * (1 / h_var * self._model.hidden.mean(x) + c / o_var * y)

        self._kernel = Normal(m, torch.sqrt(cov))

        return self._kernel.sample()

    def weight(self, y, xn, xo, *args, **kwargs):
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - self._kernel.log_prob(xn)