from torch.distributions import Distribution, AffineTransform, TransformedDistribution
import torch
from typing import Tuple
from .stochastic_process import StructuralStochasticProcess
from ..distributions import DistributionWrapper
from .typing import MeanOrScaleFun
from .state import NewState
from ..typing import ArrayType


def _define_transdist(
    loc: torch.Tensor, scale: torch.Tensor, n_dim: int, dist: Distribution
) -> TransformedDistribution:
    """
    Helper method for defining an affine transition density given the location and scale.

    Args:
        loc: The location of the distribution.
        scale: The scale of the distribution.
        n_dim: The dimension of space of the distribution.
        dist: The base distribution to apply the location-scale transform..

    Returns:
        The resulting affine transformed distribution.
    """

    loc, scale = torch.broadcast_tensors(loc, scale)

    shape = loc.shape[:-n_dim] if n_dim > 0 else loc.shape

    return TransformedDistribution(
        dist.expand(shape), AffineTransform(loc, scale, event_dim=n_dim), validate_args=False
    )


class AffineProcess(StructuralStochasticProcess):
    """
    Class for defining stochastic processes of affine nature, i.e. where we can express the next state :math:`X_{t+1}`
    given the previous state (assuming Markovian) :math:`X_t` as:
        .. math::
            X_{t+1} = f(X_t, \\theta) + g(X_t, \\theta) \\cdot W_{t+1},

    where :math:`\\theta` denotes the parameter set governing the functions :math:`f` and :math:`g`, and :math:`W_t`
    denotes random variable with arbitrary density (from which we can sample).

    Example:
        One example of an affine stochastic process is the AR(1) process. We define it by:
            >>> from pyfilter.timeseries import AffineProcess
            >>> from pyfilter.distributions import DistributionWrapper
            >>> from torch.distributions import Normal, TransformedDistribution, AffineTransform
            >>>
            >>> def f(x, alpha, beta, sigma):
            >>>     return alpha + beta * x.values
            >>>
            >>> def g(x, alpha, beta, sigma):
            >>>     return sigma
            >>>
            >>> def init_transform(model, normal_dist):
            >>>     alpha, beta, sigma = model.functional_parameters()
            >>>     return TransformedDistribution(normal_dist, AffineTransform(alpha, sigma / (1 - beta ** 2)).sqrt())
            >>>
            >>> parameters = (
            >>>     0.0,    # alpha
            >>>     0.99,   # beta
            >>>     0.05,   # sigma
            >>> )
            >>> initial_dist = increment_dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)
            >>> ar_1 = AffineProcess((f, g), parameters, initial_dist, increment_dist, initial_transform=init_transform)
            >>>
            >>> samples = ar_1.sample_path(1000)
    """

    def __init__(
        self,
        funcs: Tuple[MeanOrScaleFun, ...],
        parameters: Tuple[ArrayType, ...],
        initial_dist: DistributionWrapper,
        increment_dist: DistributionWrapper,
        **kwargs
    ):
        """
        Initializes the ``AffineProcess`` class.

        Args:
            funcs: Tuple consisting of the pair of functions ``(f, g)`` that control mean and scale respectively. Call
                signature for both is ``f(x: NewState, *parameters)``.
            parameters: See base.
            initial_dist: See base.
            increment_dist: Corresponds to the distribution that we location-scale transform.
        """

        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)
        self.f, self.g = funcs

        self.increment_dist = increment_dist

    def build_density(self, x):
        loc, scale = self.mean_scale(x)

        return _define_transdist(loc, scale, self.n_dim, self.increment_dist())

    def mean_scale(self, x: NewState, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process evaluated at ``x`` and ``.functional_parameters()`` or ``parameters``.

        Args:
            x: The previous state of the process.
            parameters: Whether to override the current parameters of the model, otherwise uses
                ``.functional_parameters()``.

        Returns:
            Returns the tuple ``(mean, scale)`` given by evaluating ``(f(x, *parameters), g(x, *parameters))``.
        """

        params = parameters or self.functional_parameters()
        return self.f(x, *params), self.g(x, *params)

    def propagate_conditional(self, x: NewState, u: torch.Tensor, parameters=None, time_increment=1.0) -> NewState:
        for _ in range(self.num_steps):
            loc, scale = self.mean_scale(x, parameters=parameters)
            x = x.propagate_from(values=loc + scale * u, time_increment=time_increment)

        return x
