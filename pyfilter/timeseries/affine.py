from .base import StochasticProcess, finite_decorator, to_state_variable
from torch.distributions import Distribution, AffineTransform, TransformedDistribution, Normal, Independent
import torch


def _get_shape(x, ndim):
    """
    Gets the shape to generate samples for.
    :param x: The tensor
    :type x: torch.Tensor|float
    :param ndim: The dimensions
    :type ndim: int
    :rtype: tuple[int]
    """

    if not isinstance(x, torch.Tensor):
        return ()

    return x.shape if ndim < 2 else x.shape[:-1]


class AffineProcess(StochasticProcess):
    def __init__(self, funcs, theta, initial_dist, increment_dist):
        """
        Class for defining model with affine dynamics.
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple[callable]
        """

        super().__init__(theta, initial_dist, increment_dist)

        # ===== Dynamics ===== #
        self.f, self.g = funcs

    def _log_prob(self, y, x):
        loc, scale = self.mean_scale(x)

        return self.predefined_weight(y, loc, scale)

    def mean_scale(self, x):
        """
        Returns the mean and scale of the process.
        :param x: The previous state
        :type x: torch.Tensor
        :return: (mean, scale)
        :rtype: tuple[torch.Tensor]
        """

        tx = to_state_variable(self, x)

        return self.f(tx, *self._theta_vals), self.g(tx, *self._theta_vals)

    @finite_decorator
    def predefined_weight(self, y, loc, scale):
        """
        Helper method for weighting with loc and scale.
        :param y: The value at x_t
        :type y: torch.Tensor
        :param loc: The mean
        :type loc: torch.Tensor
        :param scale: The scale
        :type scale: torch.Tensor
        :return: The log-weights
        :rtype: torch.Tensor
        """

        dist = self._define_transdist(loc, scale)

        return dist.log_prob(y)

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

        loc, scale = torch.broadcast_tensors(loc, scale)

        shape = _get_shape(loc, self.ndim)

        return TransformedDistribution(
            self.increment_dist.expand(shape), AffineTransform(loc, scale, event_dim=self._event_dim)
        )

    def _propagate(self, x, as_dist=False):
        dist = self._define_transdist(*self.mean_scale(x))

        if as_dist:
            return dist

        return dist.sample()


class RandomWalk(AffineProcess):
    def __init__(self, std):
        """
        Defines a random walk.
        :param std: The vector of standard deviations
        :type std: torch.Tensor|float
        """

        def f(x, s):
            return x

        def g(x, s):
            return s

        if not isinstance(std, torch.Tensor):
            normal = Normal(0., 1.)
        else:
            normal = Normal(0., 1.) if std.shape[-1] < 2 else Independent(Normal(torch.zeros_like(std), std), 1)

        super().__init__((f, g), (std,), normal, normal)
