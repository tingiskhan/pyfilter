from .base import Proposal
import torch
from typing import Tuple
from torch.distributions import Normal, MultivariateNormal, AffineTransform
from ....timeseries import LinearGaussianObservations as LGO, AffineProcess, NewState
from ....utils import construct_diag_from_flat


class LinearGaussianObservations(Proposal):
    """
    Proposal designed for cases when the observation density is a linear combination of the states, and has a Gaussian
    density. Note that your state space model must of type `LinearGaussianObservations` in order to use this proposal.
    """

    def __init__(self):
        super().__init__()
        self._hidden_is1d = None
        self._observable_is1d = None

    def set_model(self, model):
        if not isinstance(model, LGO) and not isinstance(model.hidden, AffineProcess):
            raise ValueError("Model combination not supported!")

        self._model = model
        self._hidden_is1d = self._model.hidden.n_dim == 0
        self._observable_is1d = self._model.observable.n_dim == 0

        return self

    def _kernel_1d(self, y, loc, h_var_inv, o_var_inv, c):
        cov = 1 / (h_var_inv + c ** 2 * o_var_inv)
        m = cov * (h_var_inv * loc + c * o_var_inv * y)

        kernel = Normal(m, cov.sqrt(), validate_args=False)

        return kernel

    def _kernel_2d(self, y, loc, h_var_inv, o_var_inv, c):
        tc = c if not self._observable_is1d else c.unsqueeze(-2)

        ttc = tc.transpose(-2, -1)
        diag_o_var_inv = construct_diag_from_flat(o_var_inv, self._model.observable.n_dim)
        t2 = torch.matmul(ttc, torch.matmul(diag_o_var_inv, tc))

        cov = (construct_diag_from_flat(h_var_inv, self._model.hidden.n_dim) + t2).inverse()
        t1 = h_var_inv * loc

        if self._observable_is1d:
            t2 = (diag_o_var_inv.squeeze(-1) * y).unsqueeze(-1)
        else:
            t2 = torch.matmul(diag_o_var_inv, y)

        t3 = ttc.matmul(t2)[..., 0]
        m = cov.matmul((t1 + t3).unsqueeze(-1))[..., 0]

        return MultivariateNormal(m, scale_tril=torch.cholesky(cov), validate_args=False)

    def get_constant_and_offset(self, params: Tuple[torch.Tensor, ...], x: NewState) -> (torch.Tensor, torch.Tensor):
        return params[0], None

    def sample_and_weight(self, y, x):
        new_state = self._model.hidden.propagate(x)
        affine_transform = next(trans for trans in new_state.dist.transforms if isinstance(trans, AffineTransform))

        loc, scale = affine_transform.loc, affine_transform.scale
        h_var_inv = 1 / scale ** 2

        params = self._model.observable.functional_parameters()

        new_state.values = loc
        c, offset = self.get_constant_and_offset(params, new_state)

        _, o_scale = self._model.observable.mean_scale(new_state)
        o_var_inv = 1 / o_scale ** 2

        y_offset = y - (offset if offset is not None else 0.0)
        kernel_func = self._kernel_1d if self._hidden_is1d else self._kernel_2d
        kernel = kernel_func(y_offset, loc, h_var_inv, o_var_inv, c)

        new_x = new_state.copy(new_state.dist, kernel.sample())

        return new_x, self._weight_with_kernel(y, new_x, kernel)

    def pre_weight(self, y, x):
        h_loc, h_scale = self._model.hidden.mean_scale(x)
        new_state = x.propagate_from(values=h_loc)
        o_loc, o_scale = self._model.observable.mean_scale(new_state)

        o_var = o_scale ** 2
        h_var = h_scale ** 2

        params = self._model.observable.functional_parameters()
        c, offset = self.get_constant_and_offset(params, new_state)

        if offset is not None:
            o_loc = offset

        if self._observable_is1d:
            if self._hidden_is1d:
                cov = o_var + c ** 2 * h_var
            else:
                tc = c.unsqueeze(-2)
                diag_h_var = construct_diag_from_flat(h_var, self._model.hidden.n_dim)
                cov = (o_var + tc.matmul(diag_h_var).matmul(tc.transpose(-2, -1))[..., 0, 0])

            return Normal(o_loc, cov.sqrt(), validate_args=False).log_prob(y)

        if self._hidden_is1d:
            tc = c.unsqueeze(-2)
            cov = (o_var + tc.matmul(tc.transpose(-2, -1)) * h_var)[..., 0, 0]
        else:
            diag_o_var = construct_diag_from_flat(o_var, self._model.observable.n_dim)
            diag_h_var = construct_diag_from_flat(h_var, self._model.hidden.n_dim)
            cov = diag_o_var + c.matmul(diag_h_var).matmul(c.transpose(-2, -1))

        return MultivariateNormal(o_loc, cov, validate_args=False).log_prob(y)
