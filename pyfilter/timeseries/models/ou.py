import torch
from torch.distributions import Normal, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
from ...distributions import DistributionWrapper
from ...typing import ArrayType
from ...utils import broadcast_all


def init_trans(module: "OrnsteinUhlenbeck", dist):
    kappa, gamma, sigma = module.functional_parameters()

    return TransformedDistribution(dist, AffineTransform(gamma, sigma / (2 * kappa).sqrt()))


# TODO: Should perhaps inherit from StochasticDifferentialEquation?
class OrnsteinUhlenbeck(AffineProcess):
    """
    Implements the solved Ornstein-Uhlenbeck process, i.e. the solution to the SDE
        .. math::
            dX_t = \\kappa (\\gamma - X_t) dt + \\sigma dW_t, \n
            X_0 \\sim \\mathcal{N}(\\gamma, \\frac{\\sigma}{\\sqrt{2\\kappa}},

    where :math:`\\kappa, \\sigma \\in \\mathbb{R}_+^n`, and :math:`\\gamma \\in \\mathbb{R}^n`.
    """

    def __init__(
        self,
        kappa: ArrayType,
        gamma: ArrayType,
        sigma: ArrayType,
        n_dim: int = None,
        dt: float = 1.0,
        **kwargs
    ):
        """
        Initializes the ``OrnsteinUhlenbeck`` class.

        Args:
            kappa: The reversion parameter.
            gamma: The mean parameter.
            sigma: The volatility parameter.
            n_dim: Optional parameter controlling the dimension of the process. Inferred from ``sigma`` if ``None``.
            dt: Optional, the timestep to use.
            kwargs: See base.
        """

        kappa, gamma, sigma = broadcast_all(kappa, gamma, sigma)

        if (n_dim or len(kappa.shape)) > 0:
            dist = DistributionWrapper(
                Normal, loc=torch.zeros(sigma.shape), scale=torch.ones(sigma.shape), reinterpreted_batch_ndims=1
            )
        else:
            dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        super().__init__(
            (self._f, self._g),
            (kappa, gamma, sigma),
            dist,
            dist,
            initial_transform=init_trans,
            **kwargs
        )
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _f(self, x, k, g, s):
        return g + (x.values - g) * (-k * self._dt).exp()

    def _g(self, x, k, g, s):
        return s / (2 * k).sqrt() * (1 - (-2 * k * self._dt).exp()).sqrt()
