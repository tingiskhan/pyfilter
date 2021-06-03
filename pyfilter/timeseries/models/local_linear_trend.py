import torch
from torch.distributions import Normal, Independent, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
from ...typing import ArrayType
from ...utils import concater
from ...distributions import DistributionWrapper


def mean(x, a, sigma_level, sigma_slope):
    return torch.matmul(a, x.values)


def scale(x, a, sigma_level, sigma_slope):
    return concater(sigma_level, sigma_slope)


def initial_transform(module, base_dist):
    return TransformedDistribution(base_dist, AffineTransform(0.0, concater(*module.functional_parameters()[-2:])))


class LocalLinearTrend(AffineProcess):
    """
    Implements a Local Linear Trend model.
    """

    def __init__(self, sigma_level: ArrayType, sigma_slope: ArrayType, **kwargs):
        parameters = (
            torch.tensor([
                [1.0, 1.0],
                [0.0, 1.0]
            ]),
            sigma_level,
            sigma_slope
        )

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1),
            loc=torch.zeros(2),
            scale=torch.ones(2)
        )

        super().__init__(
            (mean, scale), parameters, initial_dist, increment_dist, initial_transform=initial_transform, **kwargs
        )


def semi_mean(x, alpha, beta, sens, s_l, s_s):
    slope = x.values[..., 1]

    new_level = sens * x.values[..., 0] + slope
    new_slope = alpha * slope + beta

    return concater(new_level, new_slope)


def semi_scale(x, alpha, beta, sens, s_l, s_s):
    return scale(x, None, s_l, s_s)


class SemiLocalLinearTrend(AffineProcess):
    """
    Implements a semi local linear trend (see Tensorflow)
    """

    def __init__(self, alpha: ArrayType, beta: ArrayType, sigma_level, sigma_slope, sensitivity: ArrayType = 1.0):
        parameters = (alpha, beta, sensitivity, sigma_level, sigma_slope)

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1),
            loc=torch.zeros(2),
            scale=torch.ones(2)
        )

        super(SemiLocalLinearTrend, self).__init__(
            (semi_mean, semi_scale),
            parameters,
            initial_dist=initial_dist,
            increment_dist=increment_dist,
            initial_transform=initial_transform
        )