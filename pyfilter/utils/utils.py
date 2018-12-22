import numpy as np
from collections import Iterable
from .normalization import normalize
import torch


def get_ess(w):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :type w: torch.Tensor
    :return: The effective sample size
    :rtype: float
    """

    normalized = normalize(w)

    return normalized.sum(-1) ** 2 / (normalized ** 2).sum(-1)


def searchsorted2d(a, b):
    """
    Searches a sorted 2D array along the second axis. Basically performs a vectorized digitize. Solution provided here:
    https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy.
    :param a: The array to take the elements from
    :type a: torch.Tensor
    :param b: The indices of the elements in `a`.
    :type b: torch.Tensor
    :return: An array of indices
    :rtype: torch.Tensor
    """
    m, n = a.shape
    max_num = max(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * torch.arange(a.shape[0], dtype=max_num.dtype)[:, None]
    p = np.searchsorted((a + r).view(-1), (b + r).view(-1)).reshape(m, -1)
    return p - n * torch.arange(m, dtype=p.dtype)[:, None]


def choose(array, indices):
    """
    Function for choosing on either columns or index.
    :param array: The array to choose on
    :type array: torch.Tensor
    :param indices: The indices to choose from `array`
    :type indices: torch.Tensor
    :return: Returns the chosen elements from `array`
    :rtype: torch.Tensor
    """

    if indices.dim() < 2:
        return array[indices]

    return array[torch.arange(array.shape[-2])[:, None], indices]


def loglikelihood(w):
    """
    Calculates the estimated loglikehood given weights.
    :param w: The log weights, corresponding to likelihood
    :type w: torch.Tensor
    :return: The log-likelihood
    :rtype: torch.Tensor
    """

    maxw, _ = w.max(-1)

    axis = -1
    if maxw.dim() > 0:
        axis = 0
        w = w.t()

    return maxw + torch.log(torch.exp(w - maxw).mean(axis))


def add_dimensions(x, ndim):
    """
    Adds the required dimensions to `x`.
    :param x: The array
    :type x: torch.Tensor
    :param ndim: The dimension
    :type ndim: int
    :rtype: torch.Tensor
    """

    if not isinstance(x, torch.Tensor):
        return x

    for i in range(ndim - x.dim()):
        x = x.unsqueeze(-1)

    return x


def broadcast_all(*args):
    """
    Basically same as Torch's, but on the other axis.
    :type args: tuple[torch.Tensor]
    :rtype: tuple[torch.Tensor]
    """
    # TODO: Switch to PyTorch's when it comes

    biggest = max(a.dim() for a in args)

    out = tuple()
    for a in args:
        out += (add_dimensions(a, biggest),)

    return out


def isfinite(x):
    """
    Returns mask for finite values. Solution: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519 .
    :param x: The array
    :type x: torch.Tensor
    :return: All those that do not satisfy
    :rtype: torch.Tensor
    """

    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return not_inf & not_nan


def construct_diag(x):
    """
    Constructs a diagonal matrix based on batched data. Solution found here:
    https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    :param x: The tensor
    :type x: torch.Tensor
    :rtype: torch.Tensor
    """

    if x.dim() < 2:
        return torch.diag(x)

    b = torch.eye(x.size(1))
    c = x.unsqueeze(2).expand(*x.size(), x.size(1))

    return c * b


def resizer(tup):
    """
    Recasts all the non-array elements of an array to arrays of the same size as the array elements. If you for example
    pass an array `[[0, a], [a, 0]]`, where `a` is a numpy.ndarray, then the elements `0` will be recast into arrays
    of the same size as `a`. Implementation of
    https://stackoverflow.com/questions/47640385/create-array-from-mixed-floats-and-arrays-python
    :param tup: The array to recast.
    :type tup: Iterable
    :return: Resized array
    :rtype: np.ndarray
    """
    # TODO: Speed up
    if isinstance(tup, (int, float, np.ndarray, np.integer, np.float)):
        return tup

    asarray = np.array(tup, dtype=object)
    flat = flatten(tup)
    shape = next((e.shape for e in flat if isinstance(e, np.ndarray)), False)

    if not shape or asarray.shape[-len(shape):] == shape:
        return np.array(tup)

    out = np.empty((len(flat), *shape))
    for i, e in enumerate(flat):
        out[i] = e

    try:
        return out.reshape((*asarray.shape, *shape))
    except ValueError as e:
        raise ValueError('Most likely errors in the dimension!') from e


def flatten(iterable):
    """
    Flattens an array comprised of an arbitrary number of lists. Solution found at:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    :param iterable: The iterable you wish to flatten.
    :type iterable: collections.Iterable
    :return:
    """
    out = list()
    for el in iterable:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, np.ndarray)):
            out.extend(flatten(el))
        else:
            out.append(el)

    return out


def approx_fprime(x, f, epsilon):
    """
    Wrapper for scipy's `approx_fprime`. Handles vectorized functions.
    :param x: The point at which to approximate the gradient
    :type x: np.ndarray
    :param f: The function to approximate
    :type f: callable
    :param epsilon: The discretization to use
    :type epsilon: float
    :return: The gradient
    :rtype: np.ndarray
    """

    f0 = f(x)

    grad = np.zeros_like(x)
    ei = np.zeros_like(x)

    for k in range(x.shape[0]):
        ei[k] = 1.
        d = epsilon * ei
        grad[k] = (f(x + d) - f0) / d[k]
        ei[k] = 0.

    return grad


def line_search(f, x, p, grad, a=10, amin=1e-8):
    """
    Implements a line-search to find the scalar such that a = argmin f(x + u * p)
    :param f: The function to minimize
    :type f: callable
    :param x: Point at which to evaluate function
    :type x: np.ndarray
    :param p: The direction, basically the gradient
    :type p: np.ndarray
    :param grad: The gradient
    :type grad: np.ndarray
    :param a: The starting value for a
    :type a: float
    :param amin: The minimum value a is allowed to assume
    :type amin: float
    :return: The constant(s) that minimize the the function in doc
    :rtype: np.ndarray
    """
    # TODO: Implement full Wolfe conditions
    c = tau = 0.75

    m = (p * grad).sum(axis=0)
    t = -c * m

    fx = f(x)

    a = a * np.ones(x.shape[1:])
    if isinstance(a, np.ndarray):
        a = a[None, :]

    inds = fx - f(x + a * p) >= a * t
    while not inds.all():
        if a.ndim > 0:
            a[~inds] *= tau
        else:
            a *= tau

        inds = (fx - f(x + a * p) >= a * t) | (a <= amin)

    return a


def bfgs(f, x, epsilon=1e-7, tol=1e-2, maxiter=50):
    """
    Implements a vectorized version of the BFGS algorithm.
    :param f: The function minimize
    :type f: callable
    :param x: Starting point
    :type x: np.ndarray
    :param epsilon: The discretization to use for numerically estimated derivatives
    :type epsilon: float
    :param tol: The tolerance
    :type tol: float
    :param maxiter: The maximum number of iterations
    :type maxiter: int
    :return: The optimization results
    :rtype: OptimizeResult
    """
    if not isinstance(x, np.ndarray):
        x = np.array([x], dtype=float)

    hessinv = np.zeros((x.shape[0], *x.shape))
    hessinv[np.diag_indices(x.shape[0])] = 1

    eye = hessinv.copy()

    converged = np.zeros_like(x, dtype=bool)

    xold = x.copy()
    gradold = approx_fprime(xold, f, epsilon)

    amax = 1e2
    iters = 0
    while converged.mean() < 0.95 and iters < maxiter:
        # TODO: figure out a way to only optimize those that haven't converged. Causing errors
        p = dot(hessinv, -gradold)

        with np.errstate(divide='ignore'):
            p = p / np.sqrt((p ** 2).sum(axis=0))

        p[np.isnan(p)] = 0.
        # TODO: Seems as like it can take a too big of a step - fix this
        a = line_search(f, xold, p, gradold, a=amax)

        s = a * p
        xnew = xold + s

        gradnew = approx_fprime(xnew, f, epsilon)
        y = gradnew - gradold

        with np.errstate(divide='raise'):
            try:
                tmp = (y * s).sum(axis=0)
                rhok = 1 / tmp
            except FloatingPointError:
                if isinstance(tmp, np.ndarray):
                    tmp[tmp == 0.] = 1e-3
                else:
                    tmp = 1e-3
                rhok = 1. / tmp

        t1 = eye - s[:, None] * y[None, :] * rhok
        t2 = eye - y[:, None] * s[None, :] * rhok
        t3 = s[:, None] * s[None, :] * rhok

        hessinv = mdot(mdot(t1, hessinv), t2) + t3

        converged = np.sqrt((gradnew ** 2).sum(axis=0)) < tol

        xold = xnew.copy()
        gradold = gradnew.copy()

        amax = 2 * a.max()
        iters += 1

    return OptimizeResult(xnew, hessinv)


class OptimizeResult(object):
    def __init__(self, x, hessinv):
        self.x = x
        self.hess_inv = hessinv
