from .base import Proposal
import torch
from torch.distributions import Normal, MultivariateNormal
from ..utils import construct_diag
from .linear import LinearGaussianObservations


# TODO: Check if we can speed up
class Linearized(Proposal):
    def __init__(self):
        """
        Implements a linearized proposal using Normal distributions. Do note that this proposal should be used for
        models that are log-concave in the observation density. Otherwise `Unscented` is more suitable.
        """

        super().__init__()

    def construct(self, y, x):
        # ===== Define helpers ===== #
        mu = self._model.hidden.mean(x)
        mu.requires_grad_(True)

        oloc = self._model.observable.mean(mu)
        oscale = self._model.observable.scale(mu)

        hscale = self._model.hidden.scale(x)

        # ===== Calculate the log-likelihood ===== #
        obs_logl = self._model.observable.predefined_weight(y, oloc, oscale)
        hid_logl = self._model.hidden.predefined_weight(mu, mu, hscale)

        logl = obs_logl + hid_logl

        # ===== Do backward-pass ===== #
        oloc.backward(torch.ones_like(oloc), retain_graph=True)
        dobsx = mu.grad.clone()

        logl.backward(torch.ones_like(logl))
        dlogl = mu.grad.clone()

        mu = mu.detach()
        oscale = oscale.detach()

        if self._model.hidden_ndim < 2:
            var = 1 / (1 / hscale ** 2 + (dobsx / oscale) ** 2)
            mean = mu + var * dlogl

            self._kernel = Normal(mean, var.sqrt())

            return self
        elif self._model.observable.ndim > 1:
            raise NotImplementedError()

        h_inv_var = construct_diag(1 / hscale ** 2)
        o_inv_var = construct_diag(1 / oscale.unsqueeze(-1) ** 2)

        # TODO: Not working for multi dimensional output. Line 35 must be altered to handle Jacobian...
        temp = torch.matmul(dobsx.unsqueeze(-1), dobsx.unsqueeze(-2)) * o_inv_var
        var = (h_inv_var + temp).inverse()

        mean = mu + torch.matmul(var, dlogl.unsqueeze(-1))[..., 0]
        self._kernel = MultivariateNormal(mean, scale_tril=torch.cholesky(var))

        return self


class LocalLinearization(LinearGaussianObservations):
    def __init__(self):
        """
        Locally linearizes the observation mean function.
        """

        super().__init__()

    def set_model(self, model):
        self._model = model
        return self

    def _get_mat_and_fix_y(self, x, y):
        mu = self._model.hidden.mean(x)
        mu.requires_grad_(True)

        obs = self._model.observable.mean(mu)
        obs.backward(torch.ones_like(obs))

        mu.detach_()
        obs.detach_()
        obsdx = mu.grad

        if self._model.hidden_ndim < 2:
            ny = y - obs + obsdx * mu
        else:
            # TODO: See TODO above
            temp = torch.matmul(obsdx.unsqueeze(-2), mu.unsqueeze(-1))[..., 0]

            if self._model.obs_ndim < 2:
                temp = temp[..., 0]

            ny = y - obs + temp

        return obsdx, ny
