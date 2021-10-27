from torch.distributions import Normal, AffineTransform, TransformedDistribution
from numbers import Number
from ..affine import AffineProcess
import torch
from ...distributions import DistributionWrapper
from ...typing import ArrayType


def init_trans(module: "OrnsteinUhlenbeck", dist):
    kappa, gamma, sigma, initial = module.functional_parameters()

    initial_ = gamma if initial is None else initial

    return TransformedDistribution(dist, AffineTransform(initial_, sigma / (2 * kappa).sqrt()))


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
        initial_state_mean: ArrayType = None,
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
            initial_state_mean: Optional, whether to use another initial mean than ``gamma``.
            kwargs: See base.
        """

        if n_dim is None:
            n_dim = 0 if isinstance(sigma, Number) else len(sigma.shape)

        if n_dim > 0:
            dist = DistributionWrapper(
                Normal, loc=torch.zeros(sigma.shape), scale=torch.ones(sigma.shape), reinterpreted_batch_ndims=1
            )
        else:
            dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        super().__init__(
            (self._f, self._g),
            (kappa, gamma, sigma, initial_state_mean),
            dist,
            dist,
            initial_transform=init_trans,
            **kwargs
        )
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _f(self, x, k, g, s, _):
        return g + (x.values - g) * torch.exp(-k * self._dt)

    def _g(self, x, k, g, s, _):
        return s / (2 * k).sqrt() * (1 - torch.exp(-2 * k * self._dt)).sqrt()
