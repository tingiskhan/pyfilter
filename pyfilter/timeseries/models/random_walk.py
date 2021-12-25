from torch.distributions import Normal
import torch
from ..linear_model import LinearModel
from ...distributions import DistributionWrapper
from ...typing import ArrayType
from ...utils import broadcast_all


class RandomWalk(LinearModel):
    """
    Defines a Gaussian random walk process, i.e. in which the dynamics are given by
        .. math::
            X_{t+1} \\sim \\mathcal{N}(X_t, \\sigma), \n
            X_0 \\sim \\mathcal{N}(x_0, \\sigma),

    where :math:`x_0` is the initial mean, defaulting to zero.
    """

    def __init__(self, std: ArrayType, initial_mean: ArrayType = 0.0, initial_scale: ArrayType = 1.0, **kwargs):
        """
        Initializes the ``RandomWalk`` model.

        Args:
            std: Corresponds to :math:`\\sigma` in class doc.
            initial_mean: Optional parameter specifying the mean of the initial distribution. Defaults to 0.
            initial_scale: Optional parameter specifying the scale of the initial distribution. Defaults to 1.
            kwargs: See base.
        """

        std, initial_mean, initial_scale = broadcast_all(std, initial_mean, initial_scale)

        # TODO: Check all instead, as they might share std?
        reinterpreted_batch_ndims = None if len(std.shape) == 0 else 1
        dist = DistributionWrapper(
            Normal, loc=initial_mean, scale=initial_scale, reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )

        super().__init__(1.0, std, dist, **kwargs)
