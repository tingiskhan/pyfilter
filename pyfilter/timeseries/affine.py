from torch.distributions import Distribution, AffineTransform, TransformedDistribution, Normal, Independent
import torch
from typing import Tuple, Union
from .process import StochasticProcess
from ..distributions import DistributionWrapper
from .typing import MeanOrScaleFun
from .state import TimeseriesState


def _define_transdist(loc: torch.Tensor, scale: torch.Tensor, inc_dist: Distribution, n_dim: int):
    loc, scale = torch.broadcast_tensors(loc, scale)

    shape = loc.shape[:-n_dim] if n_dim > 0 else loc.shape

    return TransformedDistribution(inc_dist.expand(shape), AffineTransform(loc, scale, event_dim=n_dim))


class AffineProcess(StochasticProcess):
    def __init__(self, funcs: Tuple[MeanOrScaleFun, ...], parameters, initial_dist, increment_dist, **kwargs):
        """
        Class for defining model with affine dynamics. And by affine we mean affine in terms of pytorch distributions,
        that is, given a base distribution X we get a new distribution Y as
            Y = loc + scale * X

        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(parameters, initial_dist, increment_dist, **kwargs)
        self.f, self.g = funcs

    def build_density(self, x):
        loc, scale = self._mean_scale(x)

        return self._define_transdist(loc, scale)

    def mean_scale(self, x: TimeseriesState):
        """
        Returns the mean and scale of the process evaluated at x_t

        :param x: The previous state
        """

        return self._mean_scale(x)

    def _mean_scale(self, x: TimeseriesState, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return _define_transdist(loc, scale, self.increment_dist(), self.n_dim)

    def _propagate_conditional(self, x, u, parameters=None):
        loc, scale = self._mean_scale(x, parameters=parameters)
        return loc + scale * u

    def prop_apf(self, x):
        return self.propagate_state(self.f(x, *self.functional_parameters()), x)


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
