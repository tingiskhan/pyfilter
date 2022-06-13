import torch
from torch.distributions import Distribution
from typing import Callable
from abc import ABC
from .pre_weight_funcs import get_pre_weight_func
from stochproc.timeseries import StochasticProcess, StateSpaceModel, TimeseriesState


class Proposal(ABC):
    """
    Abstract base class for proposal objects.
    """

    def __init__(self, pre_weight_func: Callable[[StochasticProcess, TimeseriesState], TimeseriesState] = None):
        """
        Initializes the :class:`Proposal` object.

        Args:
            pre_weight_func: `f`unction used in the :class:`APF` when weighing the particles to be propagated. A common
                choice is :math:`p(y_t | E_t[x_{t-1}])`, where :math:`E_t[...]` denotes the expected value of the
                stochastic process at time :math:`t`` using the values at :math:`t-1`.
        """

        super().__init__()
        self._model = None  # type: StateSpaceModel
        self._pre_weight_func = pre_weight_func

    def set_model(self, model: StateSpaceModel):
        """
        Sets the model to be used in the proposal.

        Args:
            model: the model to consider.
        """

        self._model = model
        self._pre_weight_func = get_pre_weight_func(self._pre_weight_func, model.hidden)

        return self

    def _weight_with_kernel(self, y: torch.Tensor, x_dist: Distribution, x_new: TimeseriesState, kernel: Distribution) -> torch.Tensor:
        y_dist = self._model.build_density(x_new)
        return y_dist.log_prob(y) + x_dist.log_prob(x_new.values) - kernel.log_prob(x_new.values)

    def sample_and_weight(self, y: torch.Tensor, x: TimeseriesState) -> (TimeseriesState, torch.Tensor):
        """
        Method to be derived by inherited classes. Given the current observation ``y`` and previous state ``x``, this
        method samples new state values and weighs them accordingly.

        Args:
            y: the current observation.
            x: the previous state.

        Returns:
            The new state together with the associated weights.
        """

        raise NotImplementedError()

    def pre_weight(self, y: torch.Tensor, x: TimeseriesState) -> torch.Tensor:
        """
        Pre-weights previous state ``x`` w.r.t. the current observation ``y``. Used in the ``APF`` when evaluating which
        candidate particles to select for propagation.

        Args:
            y: the current observation.
            x: the previous state.

        Returns:
            Returns the log weights associated with the previous state particles.
        """

        new_state = self._pre_weight_func(self._model.hidden, x)
        y_dist = self._model.build_density(new_state)

        return y_dist.log_prob(y)

    def copy(self) -> "Proposal":
        """
        Copies the proposal by returning a new instance of type ``self``.
        """

        raise NotImplementedError()
