from .base import Proposal
from sympy import symbols, Function, lambdify
import torch
from math import sqrt
from torch.distributions import Normal, Independent

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
        f0 = f(x)
        diff = lambda u, v: (f(u + v) - f0) / h
    elif order == 2:
        diff = lambda u, v: (f(u + v) - f(u - v)) / 2 / h
    else:
        raise ValueError('Only 1st and 2nd order precision available!')

    grad = tuple()
    ei = torch.zeros_like(x)

    for k in range(x.shape[-1]):
        ei[..., k] = h

        grad += (diff(x, ei),)
        ei[..., k] = 0.

    if len(grad) < 2:
        return grad[-1][..., 0]

    return torch.cat([g.unsqueeze(-1) for g in grad], dim=-1)


# TODO: Not too fast. Try finding the bottleneck
class Linearized(Proposal):
    def __init__(self, order=2, h=eps):
        """
        Implements a linearized proposal using Normal distributions. Do note that this proposal should be used for
        models that are log-concave in the observation density. Otherwise `Unscented` is more suitable.
        :param order: The order of accuracy to use for gradient estimation
        :type order: int
        :param h: The discretization step
        :type h: float
        """

        super().__init__()

        if not (1 <= order <= 2):
            raise ValueError('Only 1st or 2nd order accuracy available!')

        self._meanexpr = get_mean_expr()
        self._order = order
        self._h = h

    def draw(self, y, x, size=None, *args, **kwargs):
        # ===== Define function ===== #
        f = lambda u: self._model.weight(y, u) + self._model.h_weight(u, x)

        # ===== Evaluate gradient ===== #
        xo = self._model.hidden.mean(x)
        grad = approx_fprime(f, xo.unsqueeze(-1) if self._model.hidden.ndim < 2 else xo, order=self._order, h=self._h)

        # ===== Get some necessary stuff ===== #
        ax = self._model.observable.mean(xo)
        bx = self._model.observable.scale(xo)

        gx = self._model.hidden.scale(x)

        # ===== Get mean ===== #
        mean = self._meanexpr(y, x, ax, bx, xo, gx, grad)

        dist = Normal(mean, gx)
        if self._model.hidden.ndim < 2:
            self._kernel = dist
        else:
            # TODO: Verify multi-dimensional models
            self._kernel = Independent(dist, x.dim() - 1)

        return self._kernel.sample()

    def weight(self, y, xn, xo, *args, **kwargs):
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - self._kernel.log_prob(xn)
