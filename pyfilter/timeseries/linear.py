import torch
from .affine import AffineProcess
from ..typing import ArrayType
from ..utils import broadcast_all


def _f0d(x, a, b, _):
    return b + a * x.values


def _f1d(x, a, b, _):
    return b + a.matmul(x.values)


def _f2d(x, a, b, _):
    return b + a.matmul(x.values.unsqueeze(-1)).squeeze(-1)


def _g(x, a, b, s):
    return s


_mapping = {0: _f0d, 1: _f1d, 2: _f2d}


class LinearModel(AffineProcess):
    """
    Implements a linear process, i.e. in which the distribution at :math:`t + 1` is given by a linear combination of
    the states at :math:`t`, i.e.
        .. math::
            X_{t+1} = b + A \\cdot X_t + \\sigma \\epsilon_{t+1}, \n
            X_0 \\sim p(x_0)
    where :math:`X_t, b, \\sigma \\in \\mathbb{R}^n`, :math:`\\{\\epsilon_t\\}` some distribution, and
    :math:`A \\in \\mathbb{R}^{n \\times n}`.
    """

    def __init__(self, a: ArrayType, sigma: ArrayType, increment_dist, b: ArrayType = None, **kwargs):
        """
        Initializes the ``LinearModel`` class.

        Args:
            a: See docs of class.
            sigma: See docs of class.
            b: See docs of class.
        """

        a = broadcast_all(a)[0]
        sigma = broadcast_all(sigma)[0]

        if b is None:
            b = torch.zeros(sigma.shape)

        dimension = len(a.shape)
        params = (a, b, sigma)

        initial_dist = kwargs.pop("initial_dist", None)

        super(LinearModel, self).__init__(
            (_mapping[dimension], _g), params, initial_dist or increment_dist, increment_dist, **kwargs
        )
