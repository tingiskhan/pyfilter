from torch.distributions import Distribution, AffineTransform, TransformedDistribution, Normal, Independent
import torch
from typing import Tuple, Union
from .stochasticprocess import StructuralStochasticProcess
from ..distributions import DistributionWrapper
from .typing import MeanOrScaleFun
from .state import NewState
from ..typing import ArrayType


def _all_are_tensors(x, y) -> bool:
    return isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)


def _define_transdist(loc: torch.Tensor, scale: torch.Tensor, n_dim: int, dist: Distribution):
    """
    Helper method for defining the transition density

    :param loc: The mean
    :param scale: The scale
    """

    if _all_are_tensors(loc, scale):
        loc, scale = torch.broadcast_tensors(loc, scale)

    shape = loc.shape[:-n_dim] if n_dim > 0 else loc.shape

    return TransformedDistribution(
        dist.expand(shape), AffineTransform(loc, scale, event_dim=n_dim), validate_args=False
    )


class AffineProcess(StructuralStochasticProcess):
    def __init__(
        self,
        funcs: Tuple[MeanOrScaleFun, ...],
        parameters: Tuple[ArrayType, ...],
        initial_dist: DistributionWrapper,
        increment_dist: DistributionWrapper,
        **kwargs
    ):
        """
        Class for defining model with affine dynamics. And by affine we mean affine in terms of pytorch distributions,
        that is, given a base distribution X we get a new distribution Y as
            Y = loc + scale * X

        :param funcs: The functions governing the dynamics of the process
        """

        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)
        self.f, self.g = funcs

        self.increment_dist = increment_dist

    def build_density(self, x):
        loc, scale = self.mean_scale(x)

        return _define_transdist(loc, scale, self.n_dim, self.increment_dist())

    def mean_scale(self, x: NewState, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process.

        :return: (mean, scale)
        """
        params = parameters or self.functional_parameters()
        return self.f(x, *params), self.g(x, *params)

    def propagate_conditional(self, x: NewState, u: torch.Tensor, parameters=None, time_increment=1.0) -> NewState:
        for _ in range(self.num_steps):
            loc, scale = self.mean_scale(x, parameters=parameters)
            x = x.propagate_from(values=loc + scale * u, time_increment=time_increment)

        return x
