import torch
from torch.distributions import Normal
from numbers import Number
from pyro.distributions import Delta
from .random_walk import RandomWalk
from ..linear import LinearModel
from ..chained import AffineChainedStochasticProcess
from ...typing import ArrayType
from ...distributions import DistributionWrapper, JointDistribution


class LocalLinearTrend(LinearModel):
    """
    Implements a Local Linear Trend model, i.e. a model with the following dynamics
        .. math::
            L_{t+1} = L_t + S_t + \\sigma_l W_{t+1}, \n
            S_{t+1} = S_t + \\sigma_s V_{t+1},

    where :math:`\\sigma_i > 0``, and :math:`W_t, V_t` are two independent zero mean and unit variance Gaussians.
    """

    def __init__(
        self,
        sigma: ArrayType,
        initial_mean: ArrayType = torch.zeros(2),
        initial_scale: ArrayType = torch.ones(2),
        **kwargs
    ):
        """
        Initializes the ``LocalLinearTrend`` class.

        Args:
            sigma: Corresponds to the vector :math:`[ \\sigma_s, \\Delta \\sigma ]``.
            initial_mean: Optional, specifies the initial mean.
            kwargs: See base.
        """

        increment_dist = DistributionWrapper(
            Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1
        )

        initial_dist = DistributionWrapper(Normal, loc=initial_mean, scale=initial_scale, reinterpreted_batch_ndims=1)

        a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        super().__init__(a, sigma, increment_dist, initial_dist=initial_dist, **kwargs)


def _smooth_dist_builder(loc, l_0, scale, validate_args=False):
    init_s = Normal(loc, scale, validate_args=validate_args)

    return JointDistribution(init_s, Delta(l_0), validate_args=validate_args)


class SmoothLinearTrend(AffineChainedStochasticProcess):
    """
    Implements a "smooth trend model", defined as
        .. math::
            L_{t+1} = L_t + S_t, \n
            S_{t+1} = S_t + \\sigma_s V_{t+1},

    see docs of ``LocalLinearTrend`` for more information regarding parameters.
    """

    def __init__(
        self,
        sigma: ArrayType,
        l_0: Number = 0.0,
        initial_mean: ArrayType = 0.0,
        initial_scale: ArrayType = 1.0,
        **kwargs
    ):
        """
        Initializes the ``SmoothLinearTrend`` class.

        Args:
            sigma: Corresponds to :math:`\\sigma_s`.
            l_0: The initial value of the level component.
            initial_mean: Optional, specifies the initial mean of the slope.
            kwargs: See base.
        """

        rw = RandomWalk(sigma, initial_mean, initial_scale, **kwargs)
        smooth = LinearModel(
            torch.ones(2), 0.0, DistributionWrapper(Delta, v=0.0), initial_dist=DistributionWrapper(Delta, v=l_0)
        )

        super(SmoothLinearTrend, self).__init__(rw=rw, smooth=smooth)
