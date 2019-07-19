from .base import Proposal
import torch
from torch.distributions import Normal, MultivariateNormal, Independent
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

        h_inv_var = construct_diag(1 / hscale ** 2)
        o_inv_var = 1 / oscale ** 2

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
            temp = torch.matmul(obsdx.unsqueeze(-2), mu.unsqueeze(-1))[..., 0]

            if self._model.obs_ndim < 2:
                temp = temp[..., 0]

            ny = y - obs + temp

        return obsdx, ny

    def weight(self, y, xn, xo):
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - self._kernel.log_prob(xn)


class ModeFinding(Proposal):
    def __init__(self, iterations=5, tol=1e-3):
        """
        Tries to find the mode of the distribution p(x_t |y_t, x_{t-1}) and then approximate the distribution using a
        normal distribution. Note that this proposal should be used in the same setting as `Linearized`, i.e. when
        the observational density is log-concave.
        :param iterations: The maximum number of iterations to perform
        :type iterations: int
        :param tol: The tolerance of gradient to quit iterating
        :type tol: float
        """

        super().__init__()
        self._iters = iterations
        self._tol = tol

    def construct(self, y, x):
        # ===== Initialize gradient ===== #
        xn = xo = self._model.hidden.mean(x)

        # TODO: Completely arbitrary starting, might not be optimal
        gamma = 0.1
        grads = tuple()
        for i in range(self._iters):
            req_grad = xo.requires_grad

            xo.requires_grad_(True)
            logl = self._model.weight(y, xo) + self._model.h_weight(xo, x)
            logl.backward(torch.ones_like(logl), retain_graph=True)

            grad = xo.grad
            xo.requires_grad_(req_grad)

            grads += (grad,)

            # ===== Calculate step size ===== #
            if len(grads) > 1:
                gdiff = grads[-1] - grads[-2]

                if self._model.hidden_ndim > 1:
                    gamma = -(((xn - xo) * gdiff).sum(-1) / (gdiff ** 2).sum(-1)).unsqueeze(-1)
                else:
                    gamma = -(xn - xo) * gdiff / gdiff ** 2

                # TODO: Perhaps use gradient info instead?
                gamma[torch.isnan(gamma)] = 0.

            xo = xn
            xn = xo + gamma * grads[-1]

            # TODO: Use while perhaps
            gradsqsum = ((grads[-1] if self._model.hidden_ndim > 1 else grads[-1].unsqueeze(-1)) ** 2).sum(-1)
            if (gradsqsum.sqrt() < self._tol).float().mean() >= 0.9:
                break

        # ===== Get distribution ====== #
        dist = Normal(xn, self._model.hidden.scale(xn))
        if self._model.hidden.ndim < 2:
            self._kernel = dist
        else:
            self._kernel = Independent(dist, 1)

        return self
