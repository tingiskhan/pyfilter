from ..affine import AffineProcess
from torch.distributions import Distribution, Normal, TransformedDistribution, AffineTransform
from ...distributions import DistributionWrapper


def _f(x, alpha, beta, sigma):
    return alpha + beta * x.values


def _g(x, alpha, beta, sigma):
    return sigma


def _init_trans(module: "AR", dist):
    alpha, beta, sigma = module.functional_parameters()
    return TransformedDistribution(dist, AffineTransform(alpha, sigma / (2 * (1 - beta)).sqrt()))


# TODO: Implement lags
class AR(AffineProcess):
    def __init__(self, alpha, beta, sigma, initial_dist: Distribution = None):
        """
        Implements a basic one dimensional autoregressive process.
        """

        inc_dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)

        super().__init__(
            (_f, _g), (alpha, beta, sigma), initial_dist or inc_dist, inc_dist, initial_transform=_init_trans
        )
