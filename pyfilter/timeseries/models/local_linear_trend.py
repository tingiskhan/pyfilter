import torch
from torch.distributions import Normal, Independent, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
from ...typing import ArrayType
from ...utils import concater
from ...distributions import DistributionWrapper


def mean(x, a, sigma2, _):
    return torch.matmul(a, x.values.unsqueeze(-1)).squeeze(-1)


def scale(x, a, sigma2, _):
    return sigma2.cumsum(-1).sqrt()


def initial_transform(module, base_dist):
    var_, initial_mean = tuple(module.functional_parameters())[-2:]
    return TransformedDistribution(base_dist, AffineTransform(initial_mean, var_.cumsum(-1).sqrt()))


class LocalLinearTrend(AffineProcess):
    """
    Implements a Local Linear Trend model.
    """

    def __init__(self, sigma2: ArrayType, initial_mean: ArrayType = torch.zeros(2), **kwargs):
        parameters = (torch.tensor([[1.0, 0.0], [1.0, 1.0]]), sigma2, initial_mean)

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2)
        )

        super().__init__(
            (mean, scale), parameters, initial_dist, increment_dist, initial_transform=initial_transform, **kwargs
        )


def semi_mean(x, mean_, coef_, sigma2, _):
    slope = x.values[..., 0]

    new_level = x.values[..., 1] + slope
    new_slope = slope + coef_ * (mean_ - slope)

    return concater(new_slope, new_level)


def semi_scale(x, alpha, beta, sigma2, _):
    return sigma2.cumsum(-1).sqrt()


def semi_initial_transform(module, base_dist):
    mean_, coef_, scale_, initial_mean = tuple(module.functional_parameters())

    scale_ = concater(scale_[..., 0] / (2 * coef_), scale_[..., 1]).cumsum(-1).sqrt()

    return TransformedDistribution(base_dist, AffineTransform(initial_mean, scale_))


class SemiLocalLinearTrend(AffineProcess):
    """
    Implements a semi local linear trend (see Tensorflow)
    """

    def __init__(
        self, mean_: ArrayType, coef_: ArrayType, sigma: ArrayType, initial_mean: ArrayType = torch.zeros(2), **kwargs
    ):
        parameters = (mean_, coef_, sigma, initial_mean)

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2)
        )

        super(SemiLocalLinearTrend, self).__init__(
            (semi_mean, semi_scale),
            parameters,
            initial_dist=initial_dist,
            increment_dist=increment_dist,
            initial_transform=semi_initial_transform,
            **kwargs
        )
