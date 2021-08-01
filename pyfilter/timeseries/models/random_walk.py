from torch.distributions import Distribution, Normal, Independent, AffineTransform, TransformedDistribution
import torch
from ..affine import AffineProcess
from ...distributions import DistributionWrapper
from ...typing import ArrayType


def _f(x, s):
    return x.values


def _g(x, s):
    return s


def init_trans(model: "RandomWalk", base_dist):
    sigma, mean = tuple(model.functional_parameters())

    return TransformedDistribution(base_dist, AffineTransform(mean, sigma))


class RandomWalk(AffineProcess):
    def __init__(self, std: ArrayType, initial_mean: ArrayType = None):
        """
        Defines a random walk.

        :param std: The vector of standard deviations
        :type std: torch.Tensor|float|Distribution
        """

        if not isinstance(std, torch.Tensor):
            normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            if std.shape[-1] < 2:
                normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
            else:
                normal = DistributionWrapper(
                    lambda **u: Independent(Normal(**u), 1), loc=torch.zeros_like(std), scale=std
                )

        initial_mean_ = initial_mean if not initial_mean is None else torch.zeros_like(std)

        super().__init__((_f, _g), (std, initial_mean_), normal, normal, initial_transform=init_trans)
