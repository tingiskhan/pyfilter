from .base import Proposal
from sympy import symbols, Function, lambdify
import torch
from math import sqrt
from torch.distributions import Normal, MultivariateNormal
from ..utils import construct_diag
from ..timeseries.statevariable import StateVariable


# TODO: Not sure if optimal, but same as `numpy`
eps = sqrt(torch.finfo(torch.float32).eps)


def get_mean_expr():
    """
    Gets the expression of the mean as a function of the parameters.
    :return: Returns the function of the mean as an expression of (y, xo, a(xo), b(xo), d(xo), g(xo), gradient)
    :rtype: callable
    """

    # ===== Define symbols and functions ====== #
    y, x, xo, grad = symbols('y x x_o grad')

    a = Function('a')
    b = Function('b')

    d = Function('d')
    g = Function('g')

    # ===== Currently only supports normal ====== #
    t1 = -(y - a(xo)) ** 2 / 2 / b(xo) ** 2
    t2 = -(x - d(xo)) ** 2 / 2 / g(xo) ** 2

    # ===== Define the linearized distribution ====== #
    args = y, xo, a(xo), b(xo), d(xo), g(xo), grad

    linearized = t1 + grad * (x - xo) + t2
    simplified = linearized.expand().collect(x)

    # ===== Get the factor prior to `x` and `x ** 2` ====== #
    fac = simplified.coeff(x, 1)
    fac2 = simplified.coeff(x, 2)

    # ===== Define the function ====== #
    return lambdify(args, -fac / fac2 / 2)


def approx_fprime(f, x, order=2, h=eps):
    """
    Approximates the derivative of `f` at `x`
    :param f: The function to differentiate
    :type f: callable
    :param x: The point at which to evaluate the derivative
    :type x: torch.Tensor
    :param order: The order of accuracy
    :type order: int
    :param h: The step size
    :type h: float
    :return: Approximation of derivative of `f` at `x`
    :rtype: torch.Tensor
    """

    if order == 1:
        f0 = f(x[..., 0] if x.shape[-1] < 2 else x)
        diff = lambda u, v: (f(u + v) - f0) / h
    elif order == 2:
        diff = lambda u, v: (f(u + v) - f(u - v)) / 2 / h
    else:
        raise ValueError('Only 1st and 2nd order precision available!')

    if x.shape[-1] < 2:
        return diff(x[..., 0], h)

    grad = torch.zeros_like(x)
    ei = torch.zeros(x.shape[-1])
    for k in range(x.shape[-1]):
        ei[k] = h

        grad[..., k] = diff(x, ei)
        ei[k] = 0.

    return grad


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
        obs_logl = self._model.observable.predefined_weight(y, x, oloc, oscale)
        hid_logl = self._model.hidden.predefined_weight(mu, x, mu, hscale)

        logl = obs_logl + hid_logl

        # ===== Do backward-pass ===== #
        oloc.backward(torch.ones_like(oloc), retain_graph=True)
        dobsx = mu.grad.clone()

        logl.backward(torch.ones_like(logl))
        dlogl = mu.grad.clone()

        mu.detach_()

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


class ModeFinding(Linearized):
    def __init__(self, iterations=5, tol=1e-3, **kwargs):
        """
        Tries to find the mode of the distribution p(x_t |y_t, x_{t-1}) and then approximate the distribution using a
        normal distribution. Note that this proposal should be used in the same setting as `Linearized`, i.e. when
        the observational density is log-concave.
        :param iterations: The maximum number of iterations to perform
        :type iterations: int
        :param tol: The tolerance of gradient to quit iterating
        :type tol: float
        :param kwargs: Any key-worded arguments passed to `Linearized`
        """

        super().__init__(**kwargs)
        self._iters = iterations
        self._tol = tol

    def construct(self, y, x):
        # ===== Define function ===== #
        f = lambda u: self._model.weight(y, u) + self._model.h_weight(u, x)

        # ===== Initialize gradient ===== #
        xn = xo = self._model.hidden.mean(x)

        # TODO: Completely arbitrary starting, might not be optimal
        gamma = 0.1
        grads = tuple()
        for i in range(self._iters):
            if self._ord is not None:
                grad = approx_fprime(
                    f, xn.unsqueeze(-1) if self._model.hidden.ndim < 2 else xn,
                    order=self._ord,
                    h=self._h
                )
            else:
                req_grad = xo.requires_grad

                xo.requires_grad_(True)
                logl = f(xo)
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
