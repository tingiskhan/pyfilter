import torch
from torch.distributions import Normal
from .model import StateSpaceModel
from .observable import LinearObservations
from ..distributions import DistributionWrapper
from ..typing import ArrayType
from ..utils import broadcast_all


class LinearSSM(StateSpaceModel):
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

        observable = LinearObservations(a, scale, base_dist)
        super().__init__(hidden, observable)


class LinearGaussianObservations(LinearSSM):
    """
    Same as ``LinearObservations`` but where the distribution :math:`W_t` is given by a Gaussian distribution with zero
    mean and unit variance.
    """

    def __init__(self, hidden, a=1.0, scale=1.0):
        """
        Initializes the ``LinearGaussianObservations`` class.

        Args:
            hidden: See base.
            a: See base.
            scale: See base.
        """

        a = broadcast_all(a)[0]
        scale = broadcast_all(scale)[0]

        if len(a.shape) == 0:
            dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            dim = a.shape[0]
            dist = DistributionWrapper(Normal, loc=torch.zeros(dim), scale=torch.ones(dim), reinterpreted_batch_ndims=1)

        super().__init__(hidden, a, scale, dist)
