from .process import StochasticProcess
from torch.distributions import Distribution, AffineTransform, TransformedDistribution, Normal, Independent
import torch
from typing import Tuple, Callable, Union
from ..distributions import DistributionWrapper


def _define_transdist(loc: torch.Tensor, scale: torch.Tensor, inc_dist: Distribution, ndim: int):
    loc, scale = torch.broadcast_tensors(loc, scale)

    shape = loc.shape[:-ndim] if ndim > 0 else loc.shape

    return TransformedDistribution(inc_dist.expand(shape), AffineTransform(loc, scale, event_dim=ndim))


MeanOrScaleFun = Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]


class AffineProcess(StochasticProcess):
    def __init__(self, funcs: Tuple[MeanOrScaleFun, ...], parameters, initial_dist, increment_dist, **kwargs):
        """
        Class for defining model with affine dynamics. And by affine we mean affine in terms of pytorch distributions,
        that is, given a base distribution X we get a new distribution Y as
            Y = loc + scale * X
        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(parameters, initial_dist, increment_dist, **kwargs)

        # ===== Dynamics ===== #
        self.f, self.g = funcs

    def define_density(self, x, u=None):
        loc, scale = self._mean_scale(x)

        return self._define_transdist(loc, scale)

    def mean_scale(self, x: torch.Tensor):
        """
        Returns the mean and scale of the process evaluated at x_t
        :param x: The previous state
        """

        return self._mean_scale(x)

    def _mean_scale(self, x: torch.Tensor, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process.
        :param x: The previous state
        :return: (mean, scale)
        """
        params = parameters or self.functional_parameters()
        return self.f(x, *params), self.g(x, *params)

    def _define_transdist(self, loc: torch.Tensor, scale: torch.Tensor):
        """
        Helper method for defining the transition density
        :param loc: The mean
        :param scale: The scale
        """

        return _define_transdist(loc, scale, self.increment_dist(), self.ndim)

    def _propagate_u(self, x, u, parameters=None):
        loc, scale = self._mean_scale(x, parameters=parameters)
        return loc + scale * u

    def prop_apf(self, x):
        return self.f(x, *self.functional_parameters())


def _f(x, s):
    return x


def _g(x, s):
    return s


class RandomWalk(AffineProcess):
    def __init__(self, std: Union[torch.Tensor, float, Distribution], initial_dist=None):
        """
        Defines a random walk.
        :param std: The vector of standard deviations
        :type std: torch.Tensor|float|Distribution
        """

        if not isinstance(std, torch.Tensor):
            normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        else:
            if std.shape[-1] < 2:
                normal = DistributionWrapper(Normal, loc=0.0, scale=1.0)
            else:
                normal = DistributionWrapper(
                    lambda **u: Independent(Normal(**u), 1), loc=torch.zeros_like(std), scale=std
                )

        super().__init__((_f, _g), (std,), initial_dist or normal, normal)
