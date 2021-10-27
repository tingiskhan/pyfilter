from torch.distributions import Normal, AffineTransform, TransformedDistribution
import torch
from numbers import Number
from ..affine import AffineProcess
from ...distributions import DistributionWrapper
from ...typing import ArrayType


def _f(x, s, _):
    return x.values


def _g(x, s, _):
    return s


def init_trans(model: "RandomWalk", base_dist):
    sigma, mean = tuple(model.functional_parameters())
    initial_mean = mean if mean is not None else base_dist.mean

    return TransformedDistribution(base_dist, AffineTransform(initial_mean, sigma))


class RandomWalk(AffineProcess):
    """
    Defines a Gaussian random walk process, i.e. in which the dynamics are given by
        .. math::
            X_{t+1} \\sim \\mathcal{N}(X_t, \\sigma), \n
            X_0 \\sim \\mathcal{N}(x_0, \\sigma),

    where :math:`x_0` is the initial mean, defaulting to zero.
    """

    def __init__(self, std: ArrayType, initial_mean: ArrayType = None, **kwargs):
        """
        Initializes the ``RandomWalk`` model.

        Args:
            std: Corresponds to :math:`\\sigma` in class doc.
            initial_mean: Optional parameter specifying the mean of the initial values. Defaults to zero if ``None``.
            kwargs: See base.
        """

        if isinstance(std, Number):
            normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            shape = std.shape

            if len(shape) == 0:
                normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
            else:
                normal = DistributionWrapper(Normal, loc=torch.zeros(shape), scale=torch.ones(shape), reinterpreted_batch_ndims=1)

        super().__init__((_f, _g), (std, initial_mean), normal, normal, initial_transform=init_trans, **kwargs)
