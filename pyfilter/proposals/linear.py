from .base import Proposal
from torch.distributions import Normal, MultivariateNormal
from ..timeseries import LinearGaussianObservations as LGO
import torch
from ..utils import construct_diag


class LinearGaussianObservations(Proposal):
    """
    Proposal designed for cases when the observation density is a linear combination of the states, and has a Gaussian
    density. Note that in order for this to work for multi-dimensional models you must use matrices to form the
    combination.
    """
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
        tc = c if c.dim() > 1 else c.unsqueeze(-2)

        # ===== Define covariance ===== #
        ttc = tc.transpose(-2, -1)
        diag_o_var_inv = construct_diag(o_var_inv if self._model.observable.ndim > 1 else o_var_inv.unsqueeze(-1))
        t2 = torch.matmul(ttc, torch.matmul(diag_o_var_inv, tc))

        cov = (construct_diag(h_var_inv) + t2).inverse()

        # ===== Get mean ===== #
        t1 = h_var_inv * loc

        t2 = diag_o_var_inv * y
        t3 = torch.matmul(ttc, t2)[..., 0]

        m = torch.matmul(cov, (t1 + t3).unsqueeze(-1))[..., 0]

        return MultivariateNormal(m, scale_tril=torch.cholesky(cov))

    def draw(self, y, x):
        # ===== Hidden ===== #
        loc = self._model.hidden.mean(x)
        h_var_inv = 1 / self._model.hidden.scale(x) ** 2

        # ===== Observable ===== #
        c = self._model.observable.theta_vals[0]
        o_var_inv = 1 / self._model.observable.scale(x) ** 2

        if self._model.hidden_ndim < 2:
            kernel = self._kernel_1d(y, loc, h_var_inv, o_var_inv, c)
        else:
            kernel = self._kernel_2d(y, loc, h_var_inv, o_var_inv, c)

        return kernel.sample()

    def weight(self, y, xn, xo):
        c = self._model.observable.theta_vals[0]
        fx = self._model.hidden.mean(xo)

        m = self._model.observable.mean(fx)

        h_var = self._model.hidden.scale(xo) ** 2
        o_var = self._model.observable.scale(xo) ** 2

        if self._model.hidden_ndim < 2:
            std = (h_var + c ** 2 * o_var).sqrt()

            return Normal(m, std).log_prob(y)

        tc = c if c.dim() > 1 else c.unsqueeze(-2)

        temp = torch.matmul(tc, torch.matmul(construct_diag(h_var), tc.transpose(-2, -1)))
        cov = construct_diag(o_var) + temp

        if self._model.obs_ndim > 1:
            return MultivariateNormal(m, scale_tril=torch.cholesky(cov)).log_prob(y)

        return Normal(m, cov[..., 0, 0].sqrt()).log_prob(y)