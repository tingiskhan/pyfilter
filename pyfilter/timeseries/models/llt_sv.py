import torch
from torch.distributions import Normal, Independent, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
from ...typing import ArrayType
from ...utils import concater
from ...distributions import DistributionWrapper


def _mean(x, reversions, means, sigmas, _):
    level = x.values[..., :2].sum(dim=-1)
    slope_and_vol = x.values[..., 1:] + reversions * (means - x.values[..., 1:])

    return concater(level, slope_and_vol[..., 0], slope_and_vol[..., 1])


def _scale(x, reversions, means, sigmas, _):
    return concater(x.values[..., -1].exp(), sigmas[..., 0], sigmas[..., 1])


def initial_transform(module, base_dist):
    reversions, means, sigmas, initial_mean = tuple(module.functional_parameters())

    mean = concater(initial_mean, means[..., 0], means[..., 1])
    scale_ = sigmas / (2 * reversions).sqrt()
    scale = concater(scale_[..., 1], scale_[..., 0], scale_[..., 1])

    return TransformedDistribution(base_dist, AffineTransform(mean, scale))


class LocalLinearTrendWithStochasticVolatility(AffineProcess):
    """
    Implements a version of local linear trend where we have stochastic volatility in the level component.
    """

    def __init__(
        self, reversions: ArrayType, means: ArrayType, sigmas: ArrayType, initial_mean: ArrayType = 0.0, **kwargs
    ):
        parameters = reversions, means, sigmas, initial_mean

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(3), scale=torch.ones(3)
        )

        super(LocalLinearTrendWithStochasticVolatility, self).__init__(
            (_mean, _scale), parameters, initial_dist, increment_dist, initial_transform=initial_transform, **kwargs
        )
