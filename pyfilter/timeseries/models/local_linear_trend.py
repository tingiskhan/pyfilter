import torch
from torch.distributions import Normal, Independent, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
from ...typing import ArrayType
from ...utils import concater
from ...distributions import DistributionWrapper


def mean(x, a, sigma, _):
    return torch.matmul(a, x.values.unsqueeze(-1)).squeeze(-1)


def scale(x, a, sigma, _):
    return sigma


def initial_transform(module, base_dist):
    scale_, initial_mean = tuple(module.functional_parameters())[-2:]
    return TransformedDistribution(base_dist, AffineTransform(initial_mean, scale_))


class LocalLinearTrend(AffineProcess):
    """
    Implements a Local Linear Trend model.
    """

    def __init__(self, sigma: ArrayType, initial_mean: ArrayType = torch.zeros(2), **kwargs):
        parameters = (torch.tensor([[1.0, 1.0], [0.0, 1.0]]), sigma, initial_mean)

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2)
        )

        super().__init__(
            (mean, scale), parameters, initial_dist, increment_dist, initial_transform=initial_transform, **kwargs
        )


def semi_mean(x, alpha, beta, sigma, _):
    slope = x.values[..., 1]

    new_level = x.values[..., 0] + slope
    new_slope = alpha + beta * slope

    return concater(new_level, new_slope)


def semi_scale(x, alpha, beta, sigma, _):
    return sigma


class SemiLocalLinearTrend(AffineProcess):
    """
    Implements a semi local linear trend (see Tensorflow)
    """

    def __init__(
        self, alpha: ArrayType, beta: ArrayType, sigma: ArrayType, initial_mean: ArrayType = torch.zeros(2), **kwargs
    ):
        parameters = (alpha, beta, sigma, initial_mean)  # TODO: Should utilize long running mean rather

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2)
        )

        super(SemiLocalLinearTrend, self).__init__(
            (semi_mean, semi_scale),
            parameters,
            initial_dist=initial_dist,
            increment_dist=increment_dist,
            initial_transform=initial_transform,
            **kwargs
        )
