from torch.distributions import Normal, TransformedDistribution, AffineTransform
from ..linear import LinearModel
from ...distributions import DistributionWrapper


def _init_trans(module: "AR", dist):
    alpha, beta, sigma = module.functional_parameters()
    return TransformedDistribution(dist, AffineTransform(alpha, sigma / (1 - beta ** 2).sqrt()))


# TODO: Implement lags
class AR(LinearModel):
    """
    Implements an AR(1) process, i.e. a process given by
        .. math::
            X_{t+1} = \\alpha + \\beta X_t + \\sigma W_t, \n
            X_0 \\sim \\mathcal{N}(\\alpha, \\frac{\\sigma}{\\sqrt{(1 - \\beta^2)})
    """

    def __init__(self, alpha, beta, sigma, **kwargs):
        """
        Initializes the ``AR`` class.

        Args:
            alpha: The mean of the process.
            beta: The reversion of the process, usually constrained to :math:`(-1, 1)`.
            sigma: The volatility of the process.
            kwargs: See base.
        """
        inc_dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        super().__init__(beta, sigma, increment_dist=inc_dist, b=alpha, initial_transform=_init_trans, **kwargs)
