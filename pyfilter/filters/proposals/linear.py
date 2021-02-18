from .base import Proposal
from torch.distributions import Normal, MultivariateNormal
from ...timeseries import LinearGaussianObservations as LGO, AffineProcess
import torch
from ...utils import construct_diag


class LinearGaussianObservations(Proposal):
    """
    Proposal designed for cases when the observation density is a linear combination of the states, and has a Gaussian
    density. Note that in order for this to work for multi-dimensional models you must use matrices to form the
    combination.
    """

    def __init__(self):
        super().__init__()
        self._mat = None

    def set_model(self, model):
        if not isinstance(model, LGO) and not isinstance(model.hidden, AffineProcess):
            raise ValueError("Model combination not supported!")

        self._model = model

        return self

    @staticmethod
    def _kernel_1d(y, loc, h_var_inv, o_var_inv, c):
        cov = 1 / (h_var_inv + c ** 2 * o_var_inv)
        m = cov * (h_var_inv * loc + c * o_var_inv * y)

        kernel = Normal(m, cov.sqrt())

        return kernel

    def _kernel_2d(self, y, loc, h_var_inv, o_var_inv, c):
        tc = c if self._model.obs_ndim > 0 else c.unsqueeze(-2)

        # ===== Define covariance ===== #
        ttc = tc.transpose(-2, -1)
        diag_o_var_inv = construct_diag(o_var_inv if self._model.observable.ndim > 0 else o_var_inv.unsqueeze(-1))
        t2 = torch.matmul(ttc, torch.matmul(diag_o_var_inv, tc))

        cov = (construct_diag(h_var_inv) + t2).inverse()

        # ===== Get mean ===== #
        t1 = h_var_inv * loc

        t2 = torch.matmul(diag_o_var_inv, y if y.dim() > 0 else y.unsqueeze(-1))
        t3 = torch.matmul(ttc, t2.unsqueeze(-1))[..., 0]

        m = torch.matmul(cov, (t1 + t3).unsqueeze(-1))[..., 0]

        return MultivariateNormal(m, scale_tril=torch.cholesky(cov))

    def sample_and_weight(self, y, x):
        # ===== Hidden ===== #
        loc, scale = self._model.hidden.mean_scale(x)
        h_var_inv = 1 / scale ** 2

        # ===== Observable ===== #
        params = self._model.observable.functional_parameters()
        c = params[0]
        o_var_inv = 1 / params[-1] ** 2

        if self._model.hidden_ndim == 0:
            kernel = self._kernel_1d(y, loc, h_var_inv, o_var_inv, c)
        else:
            kernel = self._kernel_2d(y, loc, h_var_inv, o_var_inv, c)

        new_x = kernel.sample()
        new_state = self._model.hidden.build_state(new_x, x)

        w = self._model.log_prob(y, new_state) + self._model.hidden.log_prob(new_x, x) - kernel.log_prob(new_x)

        return new_state, w

    def pre_weight(self, y, x):
        h_loc, h_scale = self._model.hidden.mean_scale(x)
        o_loc, o_scale = self._model.observable.mean_scale(self._model.hidden.build_state(h_loc, x))

        o_var = o_scale ** 2
        h_var = h_scale ** 2

        params = self._model.observable.functional_parameters()
        c = params[0]

        if self._model.obs_ndim < 1:
            if self._model.hidden_ndim < 1:
                cov = o_var + c ** 2 * h_var
            else:
                tc = c.unsqueeze(-2)
                cov = (o_var + tc.matmul(tc.transpose(-2, -1)) * h_var)[..., 0, 0]

            return Normal(o_loc, cov.sqrt()).log_prob(y)

        if self._model.hidden_ndim < 1:
            tc = c.unsqueeze(-2)
            cov = (o_var + tc.matmul(tc.transpose(-2, -1)) * h_var)[..., 0, 0]
        else:
            diag_o_var = construct_diag(o_var)
            diag_h_var = construct_diag(h_var)
            cov = diag_o_var + c.matmul(diag_h_var).matmul(c.transpose(-2, -1))

        return MultivariateNormal(o_loc, cov).log_prob(y)
