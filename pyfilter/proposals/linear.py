from .base import Proposal
from torch.distributions import Normal, MultivariateNormal
from ..timeseries import LinearGaussianObservations as LGO, AffineProcess
import torch
from ..utils import construct_diag


class LinearGaussianObservations(Proposal):
    """
    Proposal designed for cases when the observation density is a linear combination of the states, and has a Gaussian
    density. Note that in order for this to work for multi-dimensional models you must use matrices to form the
    combination.
    """

    def __init__(self):
        super().__init__()
        self._mat = None

    def _get_mat_and_fix_y(self, x, y):
        return self._model.observable._theta_vals[0], y

    def set_model(self, model):
        if not isinstance(model, LGO) and not isinstance(model.hidden, AffineProcess):
            raise ValueError('Model combination not supported!')

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

    def construct(self, y, x):
        # ===== Hidden ===== #
        loc, scale = self._model.hidden.mean_scale(x)
        h_var_inv = 1 / scale ** 2

        # ===== Observable ===== #
        c, y = self._get_mat_and_fix_y(x, y)
        o_var_inv = 1 / self._model.observable.theta_vals[-1] ** 2

        if self._model.hidden_ndim == 0:
            self._kernel = self._kernel_1d(y, loc, h_var_inv, o_var_inv, c)
        else:
            self._kernel = self._kernel_2d(y, loc, h_var_inv, o_var_inv, c)

        return self

