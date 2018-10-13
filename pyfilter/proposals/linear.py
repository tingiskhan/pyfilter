from .base import Proposal
from torch.distributions import Normal, Independent
from ..timeseries import LinearGaussianObservations as LGO


# TODO: Seems to work for 1D models currently, will need to extend to multidimensional
class LinearGaussianObservations(Proposal):
    def set_model(self, model):
        if not isinstance(model, LGO):
            raise ValueError('Model must be of instance {}'.format(LGO.__name__))

        self._model = model

        return self

    @staticmethod
    def _kernel_1d(y, loc, h_var_inv, o_var_inv, c):
        cov = 1 / (h_var_inv + c ** 2 * o_var_inv)
        m = cov * (h_var_inv * loc + c * o_var_inv * y)

        kernel = Normal(m, cov.sqrt())

        return kernel

    def _kernel_2d(self, y, loc, h_var_inv, o_var_inv, c):
        raise NotImplementedError()

    def draw(self, y, x, size=None, *args, **kwargs):
        # ===== Hidden ===== #
        loc = self._model.hidden.mean(x)
        h_var_inv = 1 / (self._model.hidden.scale(x) * self._model.hidden.noise.stddev) ** 2

        # ===== Observable ===== #
        c = self._model.observable.theta_vals[0]
        o_var_inv = 1 / (self._model.observable.theta_vals[-1] * self._model.observable.noise.stddev) ** 2

        if self._model.hidden_ndim < 2:
            self._kernel = self._kernel_1d(y, loc, h_var_inv, o_var_inv, c)
        else:
            self._kernel = self._kernel_2d(y, loc, h_var_inv, o_var_inv, c)

        return self._kernel.sample()

    def weight(self, y, xn, xo, *args, **kwargs):
        k_prob = self._kernel.log_prob(xn)

        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - k_prob