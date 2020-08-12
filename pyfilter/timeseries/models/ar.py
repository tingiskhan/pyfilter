from ..affine import AffineProcess
from torch.distributions import Distribution, Normal
from .ou import init_trans


def _f(x, alpha, beta, sigma):
    return alpha + beta * x


def _g(x, alpha, beta, sigma):
    return sigma


def _init_trans(dist, alpha, beta, sigma):
    return init_trans(dist, 1 - beta, alpha, sigma)


# TODO: Implement lags
class AR(AffineProcess):
    def __init__(self, alpha, beta, sigma, initial_dist: Distribution = None):
        """
        Implements a basic one dimensional autoregressive process.
        """

        inc_dist = Normal(0., 1.)

        super().__init__((_f, _g), (alpha, beta, sigma), initial_dist or inc_dist, inc_dist,
                         initial_transform=_init_trans)