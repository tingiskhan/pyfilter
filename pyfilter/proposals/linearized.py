from .base import Proposal
import torch
from torch.distributions import Normal, MultivariateNormal
from ..utils import construct_diag
from ..timeseries import AffineProcess


# TODO: Check if we can speed up
class Linearized(Proposal):
    def __init__(self):
        """
        Implements a linearized proposal using Normal distributions. Do note that this proposal should be used for
        models that are log-concave in the observation density. Otherwise `Unscented` is more suitable.
        """

        super().__init__()

    def set_model(self, model):
        if model.obs_ndim > 1:
            raise NotImplementedError('More observation dimensions than 1 is currently not implemented!')
        elif not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f'Both observable and hidden must be of type {AffineProcess.__class__.__name__}!')

        self._model = model

        return self

    def construct(self, y, x):
        # ===== Define helpers ===== #
        h_loc, h_scale = self._model.hidden.mean_scale(x)
        h_loc.requires_grad_(True)

        o_loc, o_scale = self._model.observable.mean_scale(h_loc)

        # ===== Calculate the log-likelihood ===== #
        obs_logl = self._model.observable.predefined_weight(y, o_loc, o_scale)
        hid_logl = self._model.hidden.predefined_weight(h_loc, h_loc, h_scale)

        logl = obs_logl + hid_logl

        # ===== Do backward-pass ===== #
        if o_loc.requires_grad:
            o_loc.backward(torch.ones_like(o_loc), retain_graph=True)
            dobsx = h_loc.grad.clone()
        else:
            dobsx = 0.

        logl.backward(torch.ones_like(logl))
        dlogl = h_loc.grad.clone()

        mu = h_loc.detach()
        oscale = o_scale.detach()
        # For cases when we return the tensor itself
        # TODO: Perhaps copy in the wrapper instead?
        x.detach_()

        if self._model.hidden_ndim < 2:
            var = 1 / (1 / h_scale ** 2 + (dobsx / oscale) ** 2)
            mean = mu + var * dlogl

            self._kernel = Normal(mean, var.sqrt())

            return self

        o_inv_var = 1 / oscale ** 2

        if self._model.observable.ndim > 1:
            ...
        else:
            o_inv_var = construct_diag(o_inv_var.unsqueeze(-1))

        h_inv_var = construct_diag(1 / h_scale ** 2)

        # TODO: Not working for multi dimensional output. Line 35 must be altered to handle Jacobian...
        temp = torch.matmul(dobsx.unsqueeze(-1), dobsx.unsqueeze(-2)) * o_inv_var
        var = (h_inv_var + temp).inverse()

        mean = mu + torch.matmul(var, dlogl.unsqueeze(-1))[..., 0]
        self._kernel = MultivariateNormal(mean, scale_tril=torch.cholesky(var))

        return self
