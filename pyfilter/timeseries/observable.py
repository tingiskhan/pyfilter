from functools import lru_cache
from abc import ABC
from torch import Size
from .stochasticprocess import StructuralStochasticProcess
from .affine import AffineProcess


class GeneralObservable(StructuralStochasticProcess, ABC):
    """
    Abstract base class constituting the observable dynamics of a state space model. Derived classes should override the
    ``.build_density(...)`` method.
    """

    def __init__(self, dimension: Size, parameters, **kwargs):
        super().__init__(parameters, initial_dist=None, **kwargs)
        self._dim = dimension

    @property
    def n_dim(self) -> int:
        return len(self._dim)

    @property
    def num_vars(self) -> int:
        return self._dim.numel()


class AffineObservations(AffineProcess):
    """
    Constitutes the observable dynamics of a state space model in which the dynamics are affine in terms of the latent
    state, i.e. we have that
        .. math::
            Y_t = f_\\theta(X_t) + g_\\theta(X_t) W_t,

    for some functions :math:`f, g` parameterized by :math:`\\theta`.
    """

    def __init__(self, funcs, parameters, increment_dist):
        """
        Initializes the ``AffineObservations`` class.

        Args:
            funcs: See base.
            parameters: See base.
            increment_dist: See base.
        """
        super().__init__(funcs, parameters, None, increment_dist)

    def initial_sample(self, shape=None):
        raise NotImplementedError("Cannot sample from Observable only!")

    @property
    @lru_cache(maxsize=None)
    def n_dim(self) -> int:
        return len(self.increment_dist().event_shape)

    @property
    @lru_cache(maxsize=None)
    def num_vars(self) -> int:
        return self.increment_dist().event_shape.numel()

    def sample_path(self, steps, **kwargs):
        raise NotImplementedError("Cannot sample from Observable only!")

    def forward(self, x, time_increment=1.0):
        return super(AffineObservations, self).forward(x, 0.0)

    propagate = forward

    def propagate_conditional(self, x, u, parameters=None, time_increment=1.0):
        return super(AffineObservations, self).propagate_conditional(x, u, parameters, 0.0)
