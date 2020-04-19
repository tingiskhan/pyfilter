from .base import StochasticProcess
from torch.distributions import Distribution, AffineTransform, TransformedDistribution, Normal, Independent
import torch


def _define_transdist(loc, scale, inc_dist, ndim):
    loc, scale = torch.broadcast_tensors(loc, scale)

    shape = loc.shape[:-ndim] if ndim > 0 else loc.shape

    return TransformedDistribution(
        inc_dist.expand(shape), AffineTransform(loc, scale, event_dim=ndim)
    )


class AffineProcess(StochasticProcess):
    def __init__(self, funcs, theta, initial_dist, increment_dist):
        """
        Class for defining model with affine dynamics. And by affine we mean affine in terms of pytorch distributions,
        that is, given a base distribution X we get a new distribution Y as
            Y = loc + scale * X
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple[callable]
        """

        super().__init__(theta, initial_dist, increment_dist)

        # ===== Dynamics ===== #
        self.f, self.g = funcs

    def _log_prob(self, y, x):
        loc, scale = self._mean_scale(x)

        return self._define_transdist(loc, scale).log_prob(y)

    def mean_scale(self, x):
        """
        Returns the mean and scale of the process evaluated at x_t
        :param x: The previous state
        :type x: torch.Tensor
        :rtype: tuple[torch.Tensor]
        """

        return self._mean_scale(x)

    def _mean_scale(self, x):
        """
        Returns the mean and scale of the process.
        :param x: The previous state
        :type x: torch.Tensor
        :return: (mean, scale)
        :rtype: tuple[torch.Tensor]
        """

        return self.f(x, *self.theta_vals), self.g(x, *self.theta_vals)

    def _define_transdist(self, loc, scale):
        """
        Helper method for defining the transition density
        :param loc: The mean
        :type loc: torch.Tensor
        :param scale: The scale
        :type scale: torch.Tensor
        :return: Distribution
        :rtype: Distribution
        """

        return _define_transdist(loc, scale, self.increment_dist, self.ndim)

    def _propagate(self, x, as_dist=False):
        dist = self._define_transdist(*self._mean_scale(x))

        if as_dist:
            return dist

        return dist.sample()

    def _propagate_u(self, x, u):
        loc, scale = self._mean_scale(x)
        return loc + scale * u


def _f(x, s):
    return x


def _g(x, s):
    return s


class RandomWalk(AffineProcess):
    def __init__(self, std):
        """
        Defines a random walk.
        :param std: The vector of standard deviations
        :type std: torch.Tensor|float|Distribution
        """

        if not isinstance(std, torch.Tensor):
            normal = Normal(0., 1.)
        else:
            normal = Normal(0., 1.) if std.shape[-1] < 2 else Independent(Normal(torch.zeros_like(std), std), 1)

        super().__init__((_f, _g), (std,), normal, normal)
