from torch.distributions import Distribution, Normal, Independent, AffineTransform, TransformedDistribution
import torch
from ..affine import AffineProcess
from ...distributions import DistributionWrapper
from ...typing import ArrayType


def _f(x, s, _):
    return x.values


def _g(x, s, _):
    return s


def init_trans(model: "RandomWalk", base_dist):
    sigma, mean = tuple(model.functional_parameters())
    initial_mean = mean if mean is not None else base_dist.loc

    return TransformedDistribution(base_dist, AffineTransform(initial_mean, sigma))


class RandomWalk(AffineProcess):
    """
    Defines a simple random walk model.
    """

    def __init__(self, std: ArrayType, initial_mean: ArrayType = None):
        if not isinstance(std, torch.Tensor):
            normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            if std.shape[-1] < 2:
                normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
            else:
                normal = DistributionWrapper(
                    lambda **u: Independent(Normal(**u), 1), loc=torch.zeros_like(std), scale=std
                )

        super().__init__((_f, _g), (std, initial_mean), normal, normal, initial_transform=init_trans)
