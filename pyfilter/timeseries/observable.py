from functools import lru_cache
from abc import ABC, abstractmethod
from torch import Size
from .stochastic_process import StructuralStochasticProcess
from .affine import AffineProcess
from .linear_model import LinearModel


class Observable(StructuralStochasticProcess, ABC):
    """
    Abstract base class for observable processes.
    """

    def _add_exog_to_state(self, x):
        # We subtract 1 as it's technically 1-indexed
        if self.exog.tensors:
            x.add_exog(self.exog[x.time_index.int() - 1])

    def initial_sample(self, shape=None):
        raise Exception("Cannot sample from Observable only!")

    def sample_path(self, steps, **kwargs):
        raise Exception("Cannot sample from Observable only!")


class GeneralObservable(Observable, ABC):
    """
    Abstract base class constituting the observable dynamics of a state space model. Derived classes should override the
    ``.build_density(...)`` method.
    """

    def __init__(self, parameters, **kwargs):
        """
        Initializes the ``GeneralObservable`` class.

        Args:
             parameters: See base.
             kwargs: See base.
        """

        super().__init__(parameters, initial_dist=None, **kwargs)

    @property
    def n_dim(self) -> int:
        return len(self.dimension)

    @property
    def num_vars(self) -> int:
        return self.dimension.numel()

    @property
    @abstractmethod
    def dimension(self) -> Size:
        """
        The dimension of the process.
        """

        pass


class Mixin(object):
    @property
    @lru_cache(maxsize=None)
    def n_dim(self) -> int:
        return len(self.increment_dist().event_shape)

    @property
    @lru_cache(maxsize=None)
    def num_vars(self) -> int:
        return self.increment_dist().event_shape.numel()

    def forward(self, x, time_increment=1.0):
        return super().forward(x, 0.0)

    propagate = forward

    def propagate_conditional(self, x, u, parameters=None, time_increment=1.0):
        return super().propagate_conditional(x, u, parameters, 0.0)


class AffineObservations(AffineProcess, Mixin, Observable):
    """
    Constitutes the observable dynamics of a state space model in which the dynamics are affine in terms of the latent
    state, i.e. we have that
        .. math::
            Y_t = f_\\theta(X_t) + g_\\theta(X_t) W_t,

    for some functions :math:`f, g` parameterized by :math:`\\theta`.
    """

    def __init__(self, funcs, parameters, increment_dist, **kwargs):
        """
        Initializes the ``AffineObservations`` class.

        Args:
            funcs: See base.
            parameters: See base.
            increment_dist: See base.
        """

        super().__init__(funcs, parameters, None, increment_dist, **kwargs)


class LinearObservations(LinearModel, Mixin, Observable):
    # TODO: Docs

    def __init__(self, a, sigma, increment_dist, **kwargs):
        """
        Initializes the ``LinearObservations`` class.

        Args:
            a: See ``LinearModel``.
            b: See ``LinearModel``.
            increment_dist: See ``LinearModel``.
        """

        super().__init__(a, sigma, increment_dist, initial_dist=None, **kwargs)
