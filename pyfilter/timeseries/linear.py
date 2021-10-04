from .model import StateSpaceModel
from .observable import AffineObservations
import torch
from torch.distributions import Distribution, Normal, Independent
from ..distributions import DistributionWrapper
from ..typing import ArrayType


def _f_0d(x, a, scale):
    return a * x.values


def _f_1d(x, a, scale):
    return _f_2d(x, a.unsqueeze(-2), scale).squeeze(-1)


def _f_2d(x, a, scale):
    return torch.matmul(a, x.values.unsqueeze(-1)).squeeze(-1)


def _g(x, a, *scale):
    return scale[-1] if len(scale) == 1 else scale


def _get_shape(a):
    is_1d = False
    if isinstance(a, Distribution):
        dim = a.event_shape
        is_1d = len(a.event_shape) == 1
    elif isinstance(a, float) or a.dim() < 2:
        dim = torch.Size([])
        is_1d = (torch.tensor(a) if isinstance(a, float) else a).dim() <= 1
    else:
        dim = a.shape[:1]

    return dim, is_1d


class LinearObservations(StateSpaceModel):
    """
    Defines a state space model where the observation dynamics are given by a linear combination of the latent states
        .. math::
            Y_t = A \\cdot X_t + \\sigma W_t,

    where :math:`A` is a matrix of size ``(dimension of observation space, dimension of latent space)``, :math:`W_t` is
    a random variable with arbitrary density, and :math:`\\sigma` is a scaling parameter.
    """

    def __init__(self, hidden, a: ArrayType, scale: ArrayType, base_dist: DistributionWrapper):
        """
        Initializes the ``LinearObservations`` class.

        Args:
            hidden: The hidden process.
            a: The matrix :math:`A`.
            scale: The scale of the.
            base_dist: The base distribution.
        """

        dim, is_1d = _get_shape(a)

        if base_dist().event_shape != dim:
            raise ValueError("The distribution is not of correct shape!")

        if not is_1d:
            f = _f_2d
        elif is_1d and hidden.n_dim > 0:
            f = _f_1d
        else:
            f = _f_0d

        observable = AffineObservations((f, _g), (a, scale), base_dist)

        super().__init__(hidden, observable)


class LinearGaussianObservations(LinearObservations):
    """
    Same as ``LinearObservations`` but where the distribution :math:`W_t` is given by a Gaussian distribution with zero
    mean and unit variance.
    """

    def __init__(self, hidden, a=1.0, scale=1.0, **kwargs):
        """
        Initializes the ``LinearGaussianObservations`` class.

        Args:
            hidden: See base.
            a: See base.
            scale: See base.
            kwargs: See base.
        """

        dim, is_1d = _get_shape(a)

        if is_1d:
            n = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            n = DistributionWrapper(
                lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(dim), scale=torch.ones(dim)
            )

        super().__init__(hidden, a, scale, n, **kwargs)
