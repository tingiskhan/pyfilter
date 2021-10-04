from ..affine import AffineProcess
from torch.distributions import Distribution, Normal, TransformedDistribution, AffineTransform
from ...distributions import DistributionWrapper


def _f(x, alpha, beta, sigma):
    return alpha + beta * x.values


def _g(x, alpha, beta, sigma):
    return sigma


def _init_trans(module: "AR", dist):
    alpha, beta, sigma = module.functional_parameters()
    return TransformedDistribution(dist, AffineTransform(alpha, sigma / ((1 - beta) ** 2.0).sqrt()))


# TODO: Implement lags
class AR(AffineProcess):
    """
    Implements an AR(1) process, i.e. a process given by
        .. math:
            X_{t+1} = \\alpha + \\beta X_t + \\sigma W_t, \n
            X_0 \\sim \\mathcal{N}(\\alpha, \\frac{\\sigma}{\\sqrt{(1 - \\beta)^2}).
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

        super().__init__(
            (_f, _g), (alpha, beta, sigma), inc_dist, inc_dist, initial_transform=_init_trans, **kwargs
        )
