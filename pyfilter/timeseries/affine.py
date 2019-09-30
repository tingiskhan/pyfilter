from .base import TimeseriesBase, init_caster, finite_decorator, tensor_caster
from torch.distributions import Distribution, AffineTransform, TransformedDistribution, Normal, Independent
import torch
from .parameter import size_getter


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


class AffineModel(TimeseriesBase):
    def __init__(self, initial, funcs, theta, noise):
        """
        Class for defining model with affine dynamics.
        :param initial: The functions governing the initial dynamics of the process
        :type initial: tuple[callable]
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple[callable]
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param noise: The noise governing the noise process
        :type noise: tuple[Distribution]
        """

        super().__init__(theta, noise)

        # ===== Dynamics ===== #
        self.f0, self.g0 = initial
        self.f, self.g = funcs

    @init_caster
    def i_mean(self):
        """
        Calculates the mean of the initial distribution.
        :return: The mean of the initial distribution
        :rtype: torch.Tensor
        """

        return self.f0(*self._theta_vals)

    @init_caster
    def i_scale(self):
        """
        Calculates the scale of the initial distribution.
        :return: The scale of the initial distribution
        :rtype: torch.Tensor
        """

        return self.g0(*self._theta_vals)

    @tensor_caster
    def f_val(self, x):
        """
        Evaluates the drift part of the process.
        :param x: The state of the process.
        :type x: torch.Tensor
        :return: The mean
        :rtype: torch.Tensor
        """

        return self.f(x, *self._theta_vals)

    @tensor_caster
    def g_val(self, x):
        """
        Evaluates the diffusion part of the process.
        :param x: The state of the process.
        :type x: torch.Tensor
        :return: The mean
        :rtype: torch.Tensor
        """

        return self.g(x, *self._theta_vals)

    def mean(self, x):
        """
        Calculates the mean of the process conditional on the previous state and current parameters.
        :param x: The state of the process.
        :type x: torch.Tensor
        :return: The mean
        :rtype: torch.Tensor
        """

        return self.f_val(x)

    def scale(self, x):
        """
        Calculates the scale of the process conditional on the current state and parameters.
        :param x: The state of the process
        :type x: torch.Tensor
        :return: The scale
        :rtype: torch.Tensor
        """

        return self.g_val(x)

    def weight(self, y, x):
        loc, scale = self.mean(x), self.scale(x)

        return self.predefined_weight(y, loc, scale)

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

        return TransformedDistribution(self.noise.expand(shape), AffineTransform(loc, scale, event_dim=self._event_dim))

    def i_sample(self, shape=None, as_dist=False):
        shape = size_getter(shape)

        dist = TransformedDistribution(
            self.noise0.expand(shape), AffineTransform(self.i_mean(), self.i_scale(), event_dim=self._event_dim)
        )

        if as_dist:
            return dist

        return dist.sample()

    def propagate(self, x, as_dist=False):
        dist = self._define_transdist(self.mean(x), self.scale(x))

        if as_dist:
            return dist

        return dist.sample()


class RandomWalk(AffineModel):
    def __init__(self, std):
        """
        Defines a random walk.
        :param std: The vector of standard deviations
        :type std: torch.Tensor|float
        """

        def f0(s):
            return torch.zeros_like(s)

        def g0(s):
            return s

        def f(x, s):
            return x

        def g(x, s):
            return s

        if not isinstance(std, torch.Tensor):
            normal = Normal(0., 1.)
        else:
            normal = Normal(0., 1.) if std.shape[-1] < 2 else Independent(Normal(torch.zeros_like(std), std), 1)

        super().__init__((f0, g0), (f, g), (std,), (normal, normal))


class AffineObservations(AffineModel):
    def __init__(self, funcs, theta, noise):
        """
        Class for defining model with affine dynamics in the observable process.
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of callable
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param noise: The noise governing the noise process
        :type noise: Distribution
        """
        super().__init__((None, None), funcs, theta, (None, noise))

    def sample(self, steps, samples=None):
        raise NotImplementedError("Cannot sample from Observable only!")

    def _verify_dimensions(self):
        # TODO: Implement this
        return self