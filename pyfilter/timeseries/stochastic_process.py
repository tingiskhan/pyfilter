from torch.distributions import Distribution
import torch
from copy import deepcopy
from typing import TypeVar, Callable, Union, Tuple, Sequence
from torch.nn import Module, Parameter
from abc import ABC
from functools import lru_cache
from .state import NewState
from ..distributions import DistributionWrapper
from ..typing import ShapeLike, ArrayType
from ..utils import size_getter
from ..prior_module import HasPriorsModule
from ..container import TensorTuple, TensorTupleMixin


T = TypeVar("T")


class StochasticProcess(TensorTupleMixin, Module, ABC):
    """
    Abstract base class for stochastic processes. By "stochastic process" we mean a sequence of random variables,
    :math:`\\{X_t\\}_{t \\in T}`, defined on a common probability space. Derived classes should override the
    ``.build_distribution(...)`` method, which builds the distribution of :math:`X_{t+1}` given
    :math:`\\{X_j\\}_{j \\leq t}`.
    """

    def __init__(
        self,
        initial_dist: DistributionWrapper,
        initial_transform: Union[Callable[["StochasticProcess", Distribution], Distribution], None] = None,
        num_steps: int = 1,
        exog: Sequence[torch.Tensor] = None,
    ):
        """
        Initializes the ``StochasticProcess`` class.

        Args:
            initial_dist: The initial distribution of the process. Corresponds to a
                ``pyfilter.distributions.DistributionWrapper`` rather than a ``pytorch`` distribution as we require
                being able to move the distribution between devices.
            initial_transform: Optional parameter allowing for re-parameterizing the initial distribution with
                parameters of the ``StochasticProcess`` object. One example is the Ornstein-Uhlenbeck process, where
                the initial distribution is usually defined as the stationary process of the distribution, which in turn
                is defined by the three parameters governing the process.
            num_steps: Optional parameter allowing to skip time steps when sampling. E.g. if we set ``num_steps`` to 5,
                we only return every fifth sample when propagating the process.
            exog: Optional parameter specifying whether to include exogenous data.
        """

        super().__init__()
        self._initial_dist = initial_dist
        self._init_transform = initial_transform
        self.num_steps = num_steps
        self.tensor_tuples["exog"] = TensorTuple(*(exog if exog is not None else ()))

    @property
    def exog(self) -> TensorTuple:
        """
        The exogenous variables.
        """

        return self.tensor_tuples["exog"]

    @property
    @lru_cache(maxsize=None)
    def n_dim(self) -> int:
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc, just like
        ``pytorch``.
        """

        return len(self.initial_dist.event_shape)

    @property
    @lru_cache(maxsize=None)
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it return the number of elements in the vector or matrix.
        """

        return self.initial_dist.event_shape.numel()

    @property
    def initial_dist(self) -> Distribution:
        """
        Returns the initial distribution and any re-parameterization given by ``._init_transform``.
        """

        dist = self._initial_dist()
        if self._init_transform is not None:
            dist = self._init_transform(self, dist)

        return dist

    def initial_sample(self, shape: ShapeLike = None) -> NewState:
        """
        Samples a state from the initial distribution.

        Args:
            shape: Optional parameter, the batch shape to use.

        Returns:
            Returns an initial sample of the process wrapped in a ``NewState`` object.
        """

        dist = self.initial_dist

        return NewState(0.0, dist.expand(size_getter(shape)))

    def build_density(self, x: NewState) -> Distribution:
        """
        Method to be overridden by derived classes. Defines how to construct the transition density to :math:`X_{t+1}`
        given the state at :math:`t`, i.e. this method corresponds to building the density:
            .. math::
                x_{t+1} \\sim p \\right ( \\cdot \\mid \\{x_j\\}_{j \\leq t} \\left ).

        Args:
            x: The previous state of the process.

        Returns:
            Returns the density of the state at :math:`t+1`.
        """

        raise NotImplementedError()

    def _add_exog_to_state(self, x: NewState):
        if self.exog.tensors:
            x.add_exog(self.exog[x.time_index.int()])

    def forward(self, x: NewState, time_increment=1.0) -> NewState:
        self._add_exog_to_state(x)

        for _ in range(self.num_steps):
            density = self.build_density(x)
            x = x.propagate_from(dist=density, time_increment=time_increment)

        return x

    def propagate(self, x: NewState, time_increment=1.0) -> NewState:
        """
        Propagates the process from a previous state to a new state. Wraps around the ``__call__`` method of
        ``pytorch.nn.Module`` to allow registering forward hooks etc.

        Args:
            x: The previous state of the process.
            time_increment: Optional parameter, the amount of time steps to increment the time index with.

        Returns:
            The new state of the process.
        """

        return self.__call__(x, time_increment=time_increment)

    def sample_path(self, steps: int, samples: ShapeLike = None, x_s: NewState = None) -> torch.Tensor:
        """
        Samples a trajectory from the stochastic process, i.e. samples the collection :math:`\\{X_j\\}_{j \\leq T}`,
        where :math:`T` corresponds to ``steps``.

        Args:
            steps: The number of steps to sample.
            samples: Optional parameter, corresponds to the batch size to sample.
            x_s: Optional parameter, whether to use a pre-defined initial state or sample a new one. If ``None`` samples
                an initial state, else uses ``x_s``.

        Returns:
            Returns a tensor of shape ``(steps, [samples], [.n_dim])``.
        """

        x_s = self.initial_sample(samples) if x_s is None else x_s

        res = (x_s,)
        for i in range(1, steps):
            res += (self.propagate(res[-1]),)

        return torch.stack(tuple(r.values for r in res), dim=0)

    def copy(self) -> "StochasticProcess":
        """
        Returns a deep copy of the object.
        """

        return deepcopy(self)

    def propagate_conditional(self, x: NewState, u: torch.Tensor, parameters=None, time_increment=1.0) -> NewState:
        """
        Propagate the process conditional on both state and draws from an incremental distribution. This method assumes
        that we may perform the following parameterization:
            .. math::
                X_{t+1} = H(t, \\{X_j\\}, W_t},

        where :math:`H: \\: T \\times \\mathcal{X}^t \\times \\mathcal{W} \\rightarrow \\mathcal{X}`, where :math:`W_t`
        are samples drawn from the incremental distribution.

        This method is mainly intended to be used filters that either require sigma points (see
        ``pyfilter.filters.kalman.UKF``), or SQMC filters (currently not implemented).

        Args:
            x: See ``.propagate(...)``
            u: The samples from the incremental distribution.
            parameters: Optional parameter, when performing the re-parameterization we sometimes require the parameters
                of ``self`` to be of another dimension (e.g. ``UKF``). This parameter allows that.
            time_increment: See ``.propagate(...)``.
        """

        self._add_exog_to_state(x)

        return

    def append_exog(self, exog: torch.Tensor):
        """
        Appends and exogenous variable.

        Args:
            exog: The new exogenous variable to add.
        """

        self.exog.append(exog)


class StructuralStochasticProcess(StochasticProcess, HasPriorsModule, ABC):
    """
    While ``StochasticProcess`` allows for any type of parameterization of ``.build_density(...)`` this derived class
    implements the special case of "classical" timeseries modelling. I.e. in which there is an analytical expression for
    the density of the next state, where the parameters comprise of the previous state and any parameters governing the
    process. An example would be the auto regressive process.
    """

    def __init__(self, parameters: Tuple[ArrayType, ...], **kwargs):
        """
        Initializes the ``StructuralStochasticProcess`` class.

        Args:
            parameters: The parameters of the analytical function, excluding the state.
            kwargs: See base.
        """

        super().__init__(**kwargs)

        for i, p in enumerate(parameters):
            self._register_parameter_or_prior(f"parameter_{i}", p)

    def functional_parameters(self, f: Callable[[torch.Tensor], torch.Tensor] = None) -> Tuple[Parameter, ...]:
        """
        Returns the functional parameters of the process, i.e. the input parameter ``parameters`` of ``.__init__(...)``.

        Args:
            f: Optional parameter, whether to apply some sort of transformation to the parameters prior to returning.
        """

        res = self.parameters_and_buffers()

        return tuple(f(v) if f is not None else v for _, v in sorted(res.items(), key=lambda k: k[0]))
